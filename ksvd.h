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
		void Init_WithRandomDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _target_sparcity, int _dictionary_size );
		/*
		 * Init solver with an initial dictionary computed by clustering the samples, and finding an optimal dictionary_size 
		 * _samples should be a preallocated buffer of _dimensionality * _sample_count size
		 */
		void Init_WithClusteredDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _target_sparcity, Scalar_t _max_cluster_error = 0.95, int _max_dictionary_size = 0 );

		/** ksvd step : train dictionary from kth sample */
		void KSVDStep( int kth );
		void OMPStep();
		void BatchOMPStep();

		/** The dictionary - nb of rows = dimensionality - nb of cols = number of atoms / dictionary size */
		Matrix_t Dict;
		/** The encoded samples - nb of rows = number of atoms / dictionary size - nb of cols = number of samples */
		Matrix_t X;
		/** The original samples - nb of rows = dimensionality - nb of cols = number of initial samples */
		Matrix_t Y;

		/** The parameters*/
		int target_sparcity;
		int dictionary_size;
		int dimensionality;
		int sample_count;
		int verbose_level;
	};

	void TestSolver();
	void SolveImg( Scalar_t* img_data, int with, int height, Scalar_t* out_data/*, Scalar_t* out_atoms, int* width_atoms, int* height_atoms*/ );

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
	target_sparcity(0),
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
void Solver::Init_WithRandomDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _target_sparcity, int _dictionary_size )
{
	target_sparcity = _target_sparcity;
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

	/*const*/ int dictionary_size = (width * height / block_size) / block_size / 4;	// dictionary atoms picked at random first

	// Random inputs
	for( int dict_idx = 0; dict_idx < dictionary_size ; dict_idx++ )
	{
		int rsample = rand() % sample_count;
		solver.Dict.col( dict_idx ) = solver.Y.col( rsample );
		solver.Dict.col( dict_idx ).normalize();	// normalize entries
	}

	Y.resize( _dimensionality, _sample_count );
	Dict.resize( _dimensionality, _dictionary_size );
	X.resize( _dictionary_size, _sample_count );
}
/*
	* Init solver with an initial dictionary computed by clustering the samples, and finding an optimal dictionary_size 
	* _samples should be a preallocated buffer of _dimensionality * _sample_count size
	*/
void Solver::Init_WithClusteredDictionary( int _dimensionality, int _sample_count, Scalar_t const* _samples, int _target_sparcity, Scalar_t _max_cluster_error, int _max_dictionary_size )
{
	target_sparcity = _target_sparcity;
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

	// From "clustering before training large datasets - case study: k-svd" paper
	//const float T_max_error = 0.95f;
	//const int fast_speed = 2;
	//const int set_size_max = 128;
	//int set_size = 16;
	int centroid_count = 1;
	IntArray_t centroid_used;
	centroid_used.push_back( 0 );
	ksvd::Matrix_t centroids( block_size, 1 );
	for( int dim_index = 0; dim_index < block_size; dim_index++ )
		centroids( dim_index, 0 ) = solver.Y( dim_index, 0 );
	centroids.col( 0 ).normalize();

	for( int sample_idx = 0; sample_idx < _sample_count; sample_idx++ )
	{
		ksvd::Vector_t sample( block_size );
		sample.col( 0 ) = solver.Y.col( sample_idx );
		sample.normalize();

		ksvd::RowVector_t centroid_dist = sample.transpose() * centroids;

		int max_idx = -1;
		float max_value = -1.f;

		// Find best matching centroid
		for( int centroid_idx = 0; centroid_idx < centroid_count; centroid_idx++ )
		{
			float dot_val = bigball::abs( centroid_dist[centroid_idx] );
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
			centroids.col( max_idx ) = centroids.col( max_idx ) * ((float)used_count / (float)(used_count + 1)) + sample.col( 0 ) / (float)(used_count + 1);
			centroid_used[max_idx]++;
		}
		else
		{
			// Add new centroid
			centroids.conservativeResize( block_size, centroid_count + 1 );
			centroids.col( centroid_count ) = sample.col( 0 );
			centroid_used.push_back( 1 );
			centroid_count++;
		}
	}

	BB_LOG( CmdTestKSVD, Log, "Found %d centroids for error %.2f", centroid_count, T_max_error );

	dictionary_size = centroid_count;
	Dict = centroids;
	X.resize( centroid_count, sample_count );
}

/*
 * ksvd step : train dictionary from kth sample 
 */
void Solver::KSVDStep( int kth )
{
	// Finw wk as the group of indices pointing to samples {yi} that use the atom dk
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
void Solver::BatchOMPStep()
{
	const Scalar_t Epsilon = (Scalar_t)1e-4;

	// Compute Graham matrix G = Dict_T.Dict
	Matrix_t Dict_T = Dict.transpose();
	Matrix_t G = Dict_T * Dict;

	const int sample_inc = (sample_count + 99) / 100;
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
#ifndef KSVD_NO_IOSTREAM
		if( verbose_level > 0 && sample_idx % (sample_inc-1) == 0 )
			std::cout << "\rOMPStep processing sample " << (sample_idx + 1) * 100 / sample_count << " %" << std::flush;
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
		//Matrix_t LTc;

		//Matrix_t dk( dimensionality, 1 );
		//Matrix_t DictI_T( 0, dimensionality );			// Incrementaly updated
		//Matrix_t xI;									// (out) -> encoded signal

		for( int k = 0; k < target_sparcity ; k++ )
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
			if( max_value < Epsilon )
				break;

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
			I_atom_count++;

			// cn = solve for c { L.LT.c = alpha0_I }
			// first solve LTc :
			Matrix_t LTc = L.triangularView<Eigen::Lower>().solve( alpha0_I );
			// then solve c :
			cn = L.transpose().triangularView<Eigen::Upper>().solve( LTc );
			
			if( k < target_sparcity-1 )
				alphan = alpha0 - (GI_T * cn);
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
void Solver::OMPStep()
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

