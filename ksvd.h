/*
	The MIT License (MIT)

	Copyright (c) 2015 nsweb

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/

/*
	Please define KSVD_IMPLEMENTATION before including this file in one C / C++ file to create the implementation.
	Define KSVD_USE_DOUBLE if you want double precision for floating point values.
*/

#include "Eigen/Dense"

/**/
namespace ksvd
{
#ifdef KSVD_USE_DOUBLE
	typedef double		Scalar_t;
#else
	typedef float		Scalar_t;
#endif

	typedef Eigen::Matrix<Scalar_t, Eigen::Dynamic, Eigen::Dynamic>						Matrix_t;
	typedef Eigen::Matrix<Scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>	MatrixRowMajor_t;
	typedef Eigen::Matrix<Scalar_t, 1, Eigen::Dynamic >									RowVector_t;
	typedef Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1>									Vector_t;
	typedef Eigen::ArrayXi																IntArray_t;

	class Solver
	{
	public:
		Solver();
		~Solver();
		
		/** Specify the sizes */
		//void SetParameters( int _target_sparcity, int _dimensionality, int _sample_count, int _dictionary_size = 0, int _verbose_level = 0 );

		/*
		 * Init solver with a random dictionary, arbitrarily taking samples as the initial atoms 
		 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
		 */
		void Init_WithRandomDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _dictionary_size );
		/*
		 * Init solver with an initial dictionary computed by clustering the samples, and finding an optimal dictionary_size 
		 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
		 */
		void Init_WithClusteredDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, Scalar_t _max_cluster_error = 0.95, int _max_dictionary_size = 0 );
		/*
		 * Augment the initial dictionary with additional centroids computed by clustering the input samples, with up to _max_dictionary_size atoms in total
		 * _samples should be a preallocated buffer of dimensionality * _sample_count size
		 */
		void AugmentDictionary( int _sample_count, Scalar_t const* _samples, Scalar_t _max_cluster_error = 0.95, int _max_dictionary_size = 0, bool fill_dict_holes = false );

		/*
		 * Compute a set of centroids approximating the input samples
		 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
		 * -> returns the number of centroids found
		 */
		int ComputeCentroids( int _dimensionality, int _sample_count, Scalar_t const* _samples, const Scalar_t T_max_error, int _max_centroid_count, Matrix_t& _out_centroids ) const;

		/** ksvd step : train dictionary from kth sample */
		void KSVDStep( int kth );
		void OMPStep( int target_sparcity );
		void BatchOMPStep( int max_sparcity, Scalar_t max_error = 0, int* sample_subset = NULL, int subset_count = 0 );

		/** The dictionary - nb of rows = dimensionality - nb of cols = number of atoms / dictionary size */
		Matrix_t Dict;
		/** The encoded samples - nb of rows = number of atoms / dictionary size - nb of cols = number of samples */
		Matrix_t X;
		/** The original samples - nb of rows = dimensionality - nb of cols = number of initial samples */
		Matrix_t Y;

		/** The parameters*/
		//int target_sparcity;
		int dictionary_size;
		int dimensionality;
		int sample_count;
		int verbose_level;
	};

	static bool IsNaN( float A ) 	{ return ((*(uint32*)&A) & 0x7FFFFFFF) > 0x7F800000; }
	void TestSolver();

}; /*namespace ksvd*/

#ifdef KSVD_IMPLEMENTATION

#include "Eigen/SVD"
#ifndef KSVD_NO_MATH
#include "math.h"
#endif
#ifndef KSVD_NO_IOSTREAM
#include <iostream>
#endif

namespace ksvd
{
Solver::Solver() :
	dictionary_size(0),
	dimensionality(0),
	sample_count(0),
	verbose_level(0)
{
		
}
Solver::~Solver()
{

}

/*
 * Init solver with a random dictionary, arbitrarily taking samples as the initial atoms 
 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
 */
void Solver::Init_WithRandomDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _dictionary_size )
{
	dictionary_size = _dictionary_size;
	dimensionality = _dimensionality;
	sample_count = _sample_count;

	// Fill Y matrix with provided samples
	Y.resize( _dimensionality, _sample_count );
	for( int sample_idx = 0; sample_idx < _sample_count; sample_idx++ )
	{
		for( int dim_index = 0; dim_index < _dimensionality; dim_index++ )
		{
			Y( dim_index, sample_idx ) = _samples[sample_idx*_dimensionality + dim_index];
		}
	}

	/*const*/ int dictionary_size = _dictionary_size ? _dictionary_size : (sample_count) / (_dimensionality * 4);	// dictionary atoms picked at random first

	// Random inputs
	for( int dict_idx = 0; dict_idx < dictionary_size ; dict_idx++ )
	{
		int rsample = rand() % sample_count;
		Dict.col( dict_idx ) = Y.col( rsample );
		Dict.col( dict_idx ).normalize();	// normalize entries
	}

	Y.resize( _dimensionality, _sample_count );
	Dict.resize( _dimensionality, _dictionary_size );
	X.resize( _dictionary_size, _sample_count );
}

/*
 * Init solver with an initial dictionary computed by clustering the samples, and finding an optimal dictionary_size 
 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
 */
void Solver::Init_WithClusteredDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, Scalar_t _max_cluster_error, int _max_dictionary_size )
{
	if( _sample_count <= 0 || !_samples )
		return;

	dictionary_size = _max_dictionary_size;
	dimensionality = _dimensionality;
	sample_count = _sample_count;

	// Fill Y matrix with provided samples
	Y.resize( _dimensionality, _sample_count );
	for( int sample_idx = 0; sample_idx < _sample_count; sample_idx++ )
	{
		for( int dim_index = 0; dim_index < _dimensionality; dim_index++ )
		{
			Y( dim_index, sample_idx ) = _samples[sample_idx*_dimensionality + dim_index];
		}
	}

	// Inspired from "clustering before training large datasets - case study: k-svd" paper
	const Scalar_t T_max_error = _max_cluster_error > (Scalar_t)0.0 ? _max_cluster_error : (Scalar_t)0.95;
	int centroid_count = ComputeCentroids( _dimensionality, _sample_count, _samples, T_max_error, _max_dictionary_size, Dict );
	dictionary_size = centroid_count;

	X.resize( centroid_count, sample_count );
}

/*
 * Augment the initial dictionary with additional centroids computed by clustering the input samples, with up to _max_dictionary_size atoms in total
 * _samples should be a preallocated buffer of dimensionality * _sample_count size
 */
void Solver::AugmentDictionary( int _sample_count, Scalar_t const* _samples, Scalar_t _max_cluster_error, int _max_dictionary_size, bool fill_dict_holes )
{
	const Scalar_t T_max_error = _max_cluster_error > (Scalar_t)0.0 ? _max_cluster_error : (Scalar_t)0.95;
	int target_additional_centroid_count = _max_dictionary_size > 0 ? _max_dictionary_size - dictionary_size : 0;
	Matrix_t additional_centroids;
	int centroid_count = ComputeCentroids( dimensionality, _sample_count, _samples, T_max_error, target_additional_centroid_count, additional_centroids );

	Dict.conservativeResize( dimensionality, dictionary_size + centroid_count );
	for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
	{
		Dict.col( dictionary_size + centroid_idx ) = additional_centroids.col( centroid_idx );
	}

	dictionary_size += centroid_count;

	X.conservativeResize( dictionary_size, sample_count );
}

/*
 * Compute a set of centroids approximating the input samples
 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
 * -> returns the number of centroids found
 */
int Solver::ComputeCentroids( int _dimensionality, int _sample_count, Scalar_t const* _samples, const Scalar_t T_max_error, int _max_centroid_count, Matrix_t& _out_centroids ) const 
{
	int centroid_count = 1;
	IntArray_t centroid_used;
	centroid_used.conservativeResize( 1 );
	centroid_used[0] = 0;

	_out_centroids.resize( _dimensionality, 1 );
	for( int dim_index = 0; dim_index < _dimensionality; dim_index++ )
		_out_centroids( dim_index, 0 ) = _samples[dim_index];//Y( dim_index, 0 );
	_out_centroids.col( 0 ).normalize();

#ifndef KSVD_NO_IOSTREAM
	if( verbose_level > 1 )
		std::cout << "New centroid " << "0 : " << _out_centroids.col( 0 ).transpose() << std::endl;
#endif

	for( int sample_idx = 0; sample_idx < _sample_count; sample_idx++ )
	{
		ksvd::Vector_t sample( _dimensionality );
		Eigen::Map<ksvd::Vector_t> mf( (Scalar_t*)(_samples + sample_idx * _dimensionality), _dimensionality, 1 );
		sample.col( 0 ) = mf;
		//sample.col( 0 ) = Y.col( sample_idx );
		if( sample.isZero( (ksvd::Scalar_t)1e-5) )
			continue;	// ignore empty samples

		sample.normalize();
		//bool bNAN = IsNaN( sample[0] );
		//if( bNAN )
		//	int Break = 0;

		//if( sample_idx % 319 == 0 && sample_idx > 57418 )
		//	std::cout << "Sample " << sample_idx << " : " << sample.transpose() << std::endl;
		ksvd::RowVector_t centroid_dist = sample.transpose() * _out_centroids;

		int max_idx = -1;
		float max_value = -1.f;

		// Find best matching centroid
		for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
		{
			float dot_val = fabs( centroid_dist[centroid_idx] );
			if( dot_val > max_value )
			{
				max_value = dot_val;
				max_idx = centroid_idx;
			}
		}

		if( max_value >= T_max_error )
		{
			// Found a good centroid candidate, average centroid pos
			int used_count = centroid_used[max_idx];
			_out_centroids.col( max_idx ) = _out_centroids.col( max_idx ) * ((float)used_count / (float)(used_count + 1)) + sample.col( 0 ) / (float)(used_count + 1);
			centroid_used[max_idx]++;

			//bNAN = IsNaN( centroids.col( max_idx )[0] );
			//if( bNAN )
			//	int Break = 0;
		}
		else
		{
			// Add new centroid
			_out_centroids.conservativeResize( _dimensionality, centroid_count + 1 );
			_out_centroids.col( centroid_count ) = sample.col( 0 );

#ifndef KSVD_NO_IOSTREAM
			if( verbose_level > 1 )
				std::cout << "New centroid " << centroid_count << " : " << _out_centroids.col( centroid_count ).transpose() << std::endl;
#endif

			centroid_used.conservativeResize( centroid_count + 1 );
			centroid_used[centroid_count] = 1;
			centroid_count++;
		}
	}

#ifndef KSVD_NO_IOSTREAM
	if( verbose_level > 0 )
		std::cout << "Found " << centroid_count << " centroids for error " << T_max_error << " in first pass" << std::endl;
#endif

	// Reduce set of centroids if necessary
	int centroids_to_remove = centroid_count - _max_centroid_count ;
	for( int remove_idx = 0; remove_idx < centroids_to_remove; remove_idx++ )
	{
		// Grab the least used centroid
		int least_used_idx = -1;
		int min_use_count = INT_MAX;
		for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
		{
			if( centroid_used[centroid_idx] > 0 && centroid_used[centroid_idx] < min_use_count )
			{
				min_use_count = centroid_used[centroid_idx];
				least_used_idx = centroid_idx;
			}
		}

		// Project that centroid into its nearest centroid
		int nearest_idx = -1;
		float nearest_value = 0.f;

		//std::cout << "centroids.col( least_used_idx ) " << centroids.col( least_used_idx ) << std::endl; 

		// Find best matching centroid
		for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
		{
			//std::cout << "centroids.col( centroid_idx ) " << centroids.col( centroid_idx ) << std::endl; 

			float dot_val = _out_centroids.col( least_used_idx ).dot( _out_centroids.col( centroid_idx ) );
			if( fabs( dot_val ) > fabs( nearest_value ) && centroid_idx != least_used_idx && centroid_used[centroid_idx] > 0 )
			{
				nearest_value = dot_val;
				nearest_idx = centroid_idx;
			}
		}

		if( nearest_value < 0.f )
			_out_centroids.col( least_used_idx ) *= -1.f;

		// Merge current centroid with best one
		float weight = (float)centroid_used[nearest_idx] / (float)(centroid_used[least_used_idx] + centroid_used[nearest_idx]);
		_out_centroids.col( nearest_idx ) = _out_centroids.col( nearest_idx ) * weight + _out_centroids.col( least_used_idx ) * (1.f - weight);
		_out_centroids.col( nearest_idx ).normalize();
		centroid_used[nearest_idx] += centroid_used[least_used_idx];
		centroid_used[least_used_idx] = 0;
	}

	// Remove unused centroids
	if( centroids_to_remove > 0 )
	{
		ksvd::Matrix_t temp_matrix( _dimensionality, _max_centroid_count );
		int centroid_push_idx = 0;
		for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
		{
			if( centroid_used[centroid_idx] > 0 )
			{
				temp_matrix.col( centroid_push_idx ) = _out_centroids.col( centroid_idx ); 
				centroid_push_idx++;
			}
		}

		_out_centroids = temp_matrix;
		centroid_count = _max_centroid_count;
	}

	return centroid_count;
}

/*
 * ksvd step : train dictionary from kth sample 
 */
void Solver::KSVDStep( int kth )
{
	// Find wk as the group of indices pointing to samples {yi} that use the atom dk
	IntArray_t wk;
	int ksample_count = 0;
	
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
		if( X(kth, sample_idx) != 0 )
		{
			wk.conservativeResize( ksample_count + 1 );
			wk[ksample_count++] = sample_idx;
		}
	}

	if( ksample_count == 0 )
		return;	// Unused atom

	// Compute Yr, which is the reduced Y that includes only the subset of samples that are currently using the atom dk
	Matrix_t Yr( dimensionality, ksample_count );
	for( int sample_idx = 0; sample_idx < ksample_count; sample_idx++ )
	{
		Yr.col( sample_idx ) = Y.col( wk[sample_idx] );
	}

	// Compute Xr, which is the reduced X that includes only the subset of samples that are currently using the atom dk
	Matrix_t Xr( dictionary_size, ksample_count );
	for( int sample_idx = 0; sample_idx < ksample_count; sample_idx++ )
	{
		Xr.col( sample_idx ) = X.col( wk[sample_idx] );
	}

	// Extract xrk
	Vector_t xrk( Xr.row(kth) );

	// Replace xrk in Xr by zeros so as to compute Erk
	Xr.row(kth).head(ksample_count).setZero();

	Matrix_t Er( Yr );
	Er -= Dict*Xr;

	// Now compute SVD of Er
	Eigen::JacobiSVD<Matrix_t> svd(Er, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// New dk is first column of U
	Vector_t dk_new = svd.matrixU().col( 0 );
	Scalar_t sing_value = svd.singularValues()( 0, 0 );
	RowVector_t xrk_new = svd.matrixV().col( 0 );
	xrk_new *= sing_value;

	Dict.col( kth ) = dk_new;

	// Update X from xrk
	for( int sample_idx = 0; sample_idx < ksample_count; sample_idx++ )
	{
		X( kth, wk[sample_idx] ) = xrk_new[sample_idx];
	}
}

/*
 * Batch Orthogonal Matching Pursuit step : optimized version of the OMPStep below, as described in the paper
 * "Efficient Implementation of the K-SVD Algorithm and the Batch-OMP Method"
 */
void Solver::BatchOMPStep( int max_sparcity, Scalar_t max_error, int* sample_subset, int subset_count )
{
	const Scalar_t Epsilon = (Scalar_t)1e-4;

	// Compute Graham matrix G = Dict_T.Dict
	Matrix_t Dict_T = Dict.transpose();
	Matrix_t G = Dict_T * Dict;

	int sample_process_count = subset_count > 0 ? subset_count : sample_count;
	const int sample_inc = (sample_process_count + 99) / 100;
	int sample_idx, iter_idx;
	for( iter_idx = 0; iter_idx < sample_process_count; iter_idx++ )
	{
		if( sample_subset )
			sample_idx = sample_subset[iter_idx];
		else
			sample_idx = iter_idx;

#ifndef KSVD_NO_IOSTREAM
		if( verbose_level > 0 && (sample_inc < 2 || sample_idx % (sample_inc-1) == 0) )
			std::cout << "\rOMPStep processing sample " << (iter_idx + 1) * 100 / sample_process_count << " %" << std::flush;
#endif

		Vector_t ysample = Y.col( sample_idx );
		Vector_t r = ysample;						// residual
		IntArray_t I_atoms;							// (out) list of selected atoms for given sample
		Matrix_t L( 1, 1 );							// Matrix from Cholesky decomposition, incrementally augmented
		L(0, 0) = (Scalar_t)1.;
		int I_atom_count = 0;
		Vector_t alpha0 = Dict_T * ysample;			// project sample on all atoms
		Vector_t alphan = alpha0;
		Vector_t alpha0_I;
		Matrix_t GI_T( dictionary_size, 0 );		// Incrementaly updated
		Matrix_t cn;
		Scalar_t delta_n = 0;
		Scalar_t error_n = ysample.dot( ysample );
		Vector_t beta;
		Vector_t beta_I;

		for( int k = 0; k < max_sparcity && error_n >= max_error; k++ )
		{
			// Select greatest component of alpha_n
			int max_idx = -1;
			Scalar_t max_value = (Scalar_t)-1.;
			for( int atom_idx = 0; atom_idx < dictionary_size; atom_idx++ )
			{
				Scalar_t dot_val = ::abs( (Scalar_t)alphan[atom_idx] );
				if( dot_val > max_value )
				{
					max_value = dot_val;
					max_idx = atom_idx;
				}
			}
			//if( max_value < Epsilon )
			//	break;

			if( I_atom_count >= 1 )
			{
				// Build column vector GI_k (in place in wM)
				Matrix_t wM( I_atom_count, 1 );		
				for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
				{
					wM(atom_idx, 0) = G( I_atoms[atom_idx], max_idx );
				}

				// w = solve for w { L.w = GI_k }
				L.triangularView<Eigen::Lower>().solveInPlace( wM );

				//            | L       0		|
				// Update L = | wT  sqrt(1-wTw)	|
				//                               
				L.conservativeResize( I_atom_count + 1, I_atom_count + 1 );
				L.row(I_atom_count).head(I_atom_count) = wM.col(0).head(I_atom_count);
				L.col(I_atom_count).setZero(); 

				Scalar_t val_tmp = 1 - wM.col(0).dot( wM.col(0) );
				L( I_atom_count, I_atom_count ) = val_tmp < 1 ? (val_tmp < 0 ? 0 : (Scalar_t) ::sqrt( (Scalar_t)val_tmp )) : 1;
			}

			I_atoms.conservativeResize( I_atom_count + 1 );
			I_atoms[I_atom_count] = max_idx;

			alpha0_I.conservativeResize( I_atom_count + 1 );
			alpha0_I[I_atom_count] = alpha0[max_idx];

			GI_T.conservativeResize( dictionary_size, I_atom_count + 1 );
			GI_T.col( I_atom_count ) = G.row( max_idx );

			// cn = solve for c { L.LT.c = alpha0_I }
			// first solve LTc :
			Matrix_t LTc = L.triangularView<Eigen::Lower>().solve( alpha0_I );
			// then solve c :
			cn = L.transpose().triangularView<Eigen::Upper>().solve( LTc );
			
			//if( k < max_sparcity-1 )
			{
				beta = GI_T * cn;
				alphan = alpha0 - beta;

				// Error based
				if( max_error > 0 )
				{
					beta_I.conservativeResize( I_atom_count + 1 );
					beta_I[I_atom_count] = beta[max_idx];

					error_n += delta_n;
					delta_n = (cn.transpose() * beta_I)(0, 0);
					error_n -= delta_n;
				}
			}

			I_atom_count++;
		}

		// Update this particular sample in X matrix
		X.col( sample_idx ).setZero();
		for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
		{
			X( I_atoms[atom_idx], sample_idx ) = cn( atom_idx, 0 );
		}
	}

#ifndef KSVD_NO_IOSTREAM
	if( verbose_level > 0 )
		std::cout << "\rOMPStep processing sample 100 %" << std::endl;
#endif
}

/*
 * Orhtogonal Matching Pursuit step : find the best projection of each sample on a given set of atoms in dictionary
 */
void Solver::OMPStep( int target_sparcity )
{
	const Scalar_t Epsilon = (Scalar_t)1e-4;

	const int sample_inc = (sample_count + 99) / 100;
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
#ifndef KSVD_NO_IOSTREAM
		if( verbose_level > 0 && sample_idx % (sample_inc-1) == 0 )
			std::cout << "\rOMPStep processing sample " << (sample_idx + 1) * 100 / sample_count << " %" << std::flush;
#endif

		Vector_t ysample = Y.col( sample_idx );
		Vector_t r = ysample;							// residual
		IntArray_t I_atoms;								// (out) list of selected atoms for given sample
		Matrix_t L( 1, 1 );								// Matrix from Cholesky decomposition, incrementally augmented
		L(0, 0) = (Scalar_t)1.;
		int I_atom_count = 0;

		Matrix_t dk( dimensionality, 1 );
		Matrix_t DictI_T( 0, dimensionality );			// Incrementaly updated
		Matrix_t xI;									// (out) -> encoded signal

		for( int k = 0; k < target_sparcity ; k++ )
		{
			// Project residual on all dictionary atoms (columns), find the one that match best
			int max_idx = -1;
			Scalar_t max_value = (Scalar_t)-1.;
			for( int atom_idx = 0; atom_idx < dictionary_size; atom_idx++ )
			{
				//std::cout << "Here is the atom " << atom_idx << " :" << Dict.col( atom_idx ) << std::endl;
				Scalar_t dot_val = ::abs( (Scalar_t)Dict.col( atom_idx ).dot( r ) );
				if( dot_val > max_value )
				{
					max_value = dot_val;
					max_idx = atom_idx;
				}
			}
			if( max_value < Epsilon )
				break;
			
			// We need to solve xI = DictI+.ysample
			// where pseudo inverse DictI+ = (DictI_T.DictI)^(-1).DictI_T
			// so xI = (DictI_T.DictI)^(-1).alpha_I where alpha_I = DictI_T.ysample

			if( I_atom_count >= 1 )
			{
				dk.col( 0 ) = Dict.col( max_idx );
				Matrix_t DITdk = DictI_T * dk;

				// w = solve for w { L.w = DictIT.dk }
				Matrix_t w = L.triangularView<Eigen::Lower>().solve( DITdk );
					
				//            | L       0		|
				// Update L = | wT  sqrt(1-wTw)	|
				//                               
				L.conservativeResize( I_atom_count + 1, I_atom_count + 1 );
				L.row(I_atom_count).head(I_atom_count) = w.col(0).head(I_atom_count);
				L.col(I_atom_count).setZero(); 

				Scalar_t val_tmp = 1 - w.col(0).dot( w.col(0) );
				L( I_atom_count, I_atom_count ) = val_tmp < 1 ? (val_tmp < 0 ? 0 : (Scalar_t) ::sqrt( (Scalar_t)val_tmp )) : 1;
			}

			I_atoms.conservativeResize( I_atom_count + 1 );
			I_atoms[I_atom_count] = max_idx;

			DictI_T.conservativeResize( I_atom_count + 1, dimensionality );
			DictI_T.row( I_atom_count ) = Dict.col( max_idx );
			I_atom_count++;

			Matrix_t alpha_I( I_atom_count, 1 );
			alpha_I = DictI_T * ysample;

			// xI = solve for c { L.LT.c = alpha_I }
			// first solve LTc :
			Matrix_t LTc = L.triangularView<Eigen::Lower>().solve( alpha_I );
			// then solve xI :
			xI = L.transpose().triangularView<Eigen::Upper>().solve( LTc );

			// r = y - Dict_I * xI
			r = ysample - DictI_T.transpose() * xI;
		}

		// Update this particular sample in X matrix
		X.col( sample_idx ).setZero();
		for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
		{
			X( I_atoms[atom_idx], sample_idx ) = xI( atom_idx, 0 );
		}
	}

#ifndef KSVD_NO_IOSTREAM
	if( verbose_level > 0 )
		std::cout << "\rOMPStep processing sample 100 %" << std::endl;
#endif
}

#if 0
void TestSolver()
{
	Solver solver;
	solver.Init( 2 /*target_sparcity*/, 4 /*dictionary_size*/, 2 /*dimensionality*/, 16 /*sample_count*/ );
	
	// Fill Y matrix, which represents the original samples
	//Matrix_t Y( solver.dimensionality, solver.sample_count );
	for( int group_idx = 0; group_idx < 4 ; group_idx++ )
	{
		Scalar_t group_x = (Scalar_t)-0.5 + (Scalar_t)(group_idx % 2);
		Scalar_t group_y = (Scalar_t)-0.5 + (Scalar_t)(group_idx / 2);

		for( int sub_group_idx = 0; sub_group_idx < 4 ; sub_group_idx++ )
		{
			Scalar_t sub_group_x = group_x - (Scalar_t)0.1 + (Scalar_t)0.2*(sub_group_idx % 2);
			Scalar_t sub_group_y = group_y - (Scalar_t)0.1 + (Scalar_t)0.2*(sub_group_idx / 2);

			int sample_idx = 4*group_idx + sub_group_idx;
			solver.Y(0, sample_idx) = sub_group_x;
			solver.Y(1, sample_idx) = sub_group_y;
		}
	}
	std::cout << "Here is the matrix Y:" << std::endl << solver.Y << std::endl;

	// Initial dictionnary
	const Scalar_t Sqrt2 = (Scalar_t)(sqrt( 2.0 ) * 0.5);
	//Matrix_t Dict( solver.dimensionality, solver.dictionary_size );
	solver.Dict(0, 0) = -Sqrt2;
	solver.Dict(1, 0) = -Sqrt2;
	solver.Dict(0, 1) = Sqrt2;
	solver.Dict(1, 1) = -Sqrt2;
	solver.Dict(0, 2) = -Sqrt2;
	solver.Dict(1, 2) = Sqrt2;
	solver.Dict(0, 3) = Sqrt2;
	solver.Dict(1, 3) = Sqrt2;
	std::cout << "Here is the matrix Dict:" << std::endl << solver.Dict << std::endl;

	// Init X
	for( int sample_idx = 0; sample_idx < 16; sample_idx++ )
	{
		for( int i=0; i<4; i++ )
			solver.X( i, sample_idx ) = (Scalar_t)((i == (sample_idx / 4)) ? 1 : 0);
	}

	std::cout << "Here is the matrix X:" << std::endl << solver.X << std::endl;

	// Encoded signal
	//Matrix_t X( solver.dictionary_size, solver.sample_count );

	for( int kth = 0; kth < solver.dictionary_size ; kth++ )
	{
		std::cout << "ksvd step: " << kth << std::endl;
		solver.KSVDStep( kth );
	}

	solver.OMPStep();

	std::cout << "Here is the matrix Dict*X:" << std::endl << solver.Dict*solver.X << std::endl;
	std::cout << "Compare with matrix Y:" << std::endl << solver.Y << std::endl;
}
#endif // 0

}; /*namespace ksvd*/

#endif // KSVD_IMPLEMENTATION

