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

#include "ksvd.h"
#include "Eigen/SVD"
#include <cmath>
#include <iostream>



namespace ksvd
{

Solver::Solver() :
	target_sparcity(0),
	dictionary_size(0),
	dimensionality(0),
	sample_count(0)
{
		
}
Solver::~Solver()
{

}

void Solver::Init( int _target_sparcity, int _dictionary_size, int _dimensionality, int _sample_count )
{
	target_sparcity = _target_sparcity;
	dictionary_size = _dictionary_size;
	dimensionality = _dimensionality;
	sample_count = _sample_count;

	Y.resize( _dimensionality, _sample_count );
	Dict.resize( _dimensionality, _dictionary_size );
	X.resize( _dictionary_size, _sample_count );
}

void Solver::KSVDStep( int kth )
{
	// Finw wk as the group of indices pointing to samples {yi} that use the atom dk
	ARRAY_T(int) wk;
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
		if( X(kth, sample_idx) != 0 )
		{
			PUSH_ARRAY_T( wk, sample_idx );
		}
	}

	int ksample_count = SIZE_ARRAY_T(wk);

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
	Vector_t xrk( Xr.row( kth ) );

	// Replace xrk in Xr by zeros so as to compute Erk
	for( int sample_idx = 0; sample_idx < ksample_count; sample_idx++ )
	{
		Xr( kth, sample_idx ) = 0;
	}
	
	Matrix_t Er( Yr );
	Er -= Dict*Xr;

	// Now compute SVD of Er
	Eigen::JacobiSVD<Matrix_t> svd(Er, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// New dk is first column of U
	Vector_t dk_new = svd.matrixU().col( 0 );
	Scalar_t sing_value = svd.singularValues()( 0, 0 );
	Vector_t xrk_new = svd.matrixV().col( 0 );
	xrk_new *= sing_value;

	Dict.col( kth ) = dk_new;
	std::cout << "Here is the matrix Dict:" << std::endl << Dict << std::endl;

	// Update X from xrk
	for( int sample_idx = 0; sample_idx < ksample_count; sample_idx++ )
	{
		X( kth, wk[sample_idx] ) = xrk_new[sample_idx];
	}
	std::cout << "Here is the matrix X:" << std::endl << X << std::endl;

	//Eigen::Matrix<Scalar_t, dimensionality, ksample_count, Eigen::ColMajor> Yr;
	
	//Matrix_t Omega( sample_count, ksample_count );
	//Matrix_t Xr = X * Omega;
	//Matrix_t Yr = Y * Omega;
	

	// Compute reduce matrix Dk with column Dk set to 0
	//Matrix_t Dk( Dict );
	//for( int dim_idx = 0; dim_idx < dimensionality; dim_idx++ )
	//{
	//	Dk( dim_idx, kth ) = 0;
	//}

	//Matrix_t Ek = Y - Dk * X;
	//Matrix_t Er = Ek * Omega;
}



void Solver::BatchOMPStep()
{
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
		Vector_t ysample = Y.col( sample_idx );
		Vector_t r = ysample;							// residual
		ARRAY_T(int) I_atoms;							// (out) list of selected atoms for given sample
		Matrix_t L( 1, 1 );								// Matrix from Cholesky decomposition, incrementally augmented
		L(0, 0) = (Scalar_t)1.;
		int I_atom_count = 0;
		Matrix_t G(dictionary_size, dictionary_size);	// G = DTD (fat matrix!)
		G = Dict.transpose() * Dict;

		Matrix_t alpha_0 = Dict.transpose() * ysample;
		Matrix_t alpha = alpha_0;

		//Matrix_t dk( dimensionality, 1 );

		for( int iter = 0; iter < target_sparcity ; iter++ )
		{
			// Project residual on all dictionary atoms, find the one that match best
			int max_idx = -1;
			Scalar_t max_value = (Scalar_t)-1.;
			for( int atom_idx = 0; atom_idx < dictionary_size; atom_idx++ )
			{
				Scalar_t dot_val = abs( alpha( atom_idx, 0 ) ); 
				if( dot_val > max_value )
				{
					// Ensure atom was not already selected previously
					int I_idx = 0;
					for( ; I_idx < SIZE_ARRAY_T(I_atoms); I_idx++ )
					{
						if( atom_idx == I_atoms[I_idx] )
							break;
					}

					if( I_idx >= SIZE_ARRAY_T(I_atoms) )
					{
						max_value = dot_val;
						max_idx = atom_idx;
					}
				}
			}
			if( max_idx != -1 )
			{
				PUSH_ARRAY_T( I_atoms, max_idx );
				I_atom_count++;

	// Fill partial dictionary matrix with only selected atoms
	//Matrix_t DictI_T( I_atom_count, dimensionality );
	//for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
	//{
	//	DictI_T.row( atom_idx ) = Dict.col( I_atoms[atom_idx] );
	//}

	//dk.col( 0 ) = Dict.col( I_atoms[I_atom_count-1] );

	//// w = solve for w { L.w = DictIT.dk }
	//Matrix_t DITdk = DictI_T * dk;
	//std::cout << "Here is the matrix DITdk:" << DITdk << std::endl;

	//			Matrix_t w( ;



	//			      Matrix<real_t, iteration, 1> w;
 // 
 //     for(size_t i = 0; i < iteration; ++i)
	//w[i] = G(indices[i], k);



				// Fill w with GI,max_idx

				//if( I_atom_count == 1 )
				//{
				//	w = DITdk;
				//}
				//else
				//{
				//	w = L.triangularView<Eigen::Lower>().solve( DITdk );
				//}

				////            | L       0		|
				//// Update L = | wT  sqrt(1-wTw)	|
				////                               
				//L.conservativeResize( I_atom_count + 1, I_atom_count + 1 );
				//L.row(I_atom_count).head(I_atom_count) = w.col(0).head(I_atom_count);
				//L.col(I_atom_count).setZero(); 

				////for (int i = 0; i < I_atom_count; i++)
				////{
				////	L( I_atom_count, i ) = w( i, 0 );
				////	L( i, I_atom_count ) = 0;
				////}
				//L( I_atom_count, I_atom_count ) = (Scalar_t) sqrt( (Scalar_t)1. - w.col(0).dot( w.col(0) ) );

				//std::cout << "Here is the matrix L:" << L << std::endl;


				// xI = solve for c { L.LT.c = xI }


			}
		}

	}
}

void Solver::OMPStep()
{
	const Scalar_t Epsilon = (Scalar_t)1e-4;

	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++ )
	{
		Vector_t ysample = Y.col( sample_idx );
		Vector_t r = ysample;							// residual
		ARRAY_T(int) I_atoms;							// (out) list of selected atoms for given sample
		Matrix_t L( 1, 1 );								// Matrix from Cholesky decomposition, incrementally augmented
		L(0, 0) = (Scalar_t)1.;
		int I_atom_count = 0;

		Matrix_t dk( dimensionality, 1 );
		Matrix_t DictI_T( 0, dimensionality );			// Incrementaly updated
		Matrix_t xI;									// (out) -> encoded signal

		for( int k = 0; k < target_sparcity ; k++ )
		{
			// Project residual on all dictionary atoms, find the one that match best
			int max_idx = -1;
			Scalar_t max_value = (Scalar_t)-1.;
			for( int atom_idx = 0; atom_idx < dictionary_size; atom_idx++ )
			{
				Scalar_t dot_val = abs( Dict.col( atom_idx ).dot( r ) );
				if( dot_val > max_value )
				{
					// Ensure atom was not already selected previously
					int I_idx = 0;
					for( ; I_idx < SIZE_ARRAY_T(I_atoms); I_idx++ )
					{
						if( atom_idx == I_atoms[I_idx] )
							break;
					}

					if( I_idx >= SIZE_ARRAY_T(I_atoms) )
					{
						max_value = dot_val;
						max_idx = atom_idx;
					}
				}
			}
			if( max_idx != -1 && max_value > Epsilon )
			{
				if( I_atom_count >= 1 )
				{
					// Fill partial dictionary matrix with only selected atoms
					//Matrix_t DictI_T( I_atom_count, dimensionality );
					//for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
					//{
					//	DictI_T.row( atom_idx ) = Dict.col( I_atoms[atom_idx] );
					//}

					dk.col( 0 ) = Dict.col( max_idx );

					Matrix_t DITdk = DictI_T * dk;
					//std::cout << "Here is the matrix DITdk:" << DITdk << std::endl;

					// w = solve for w { L.w = DictIT.dk }
					Matrix_t w = L.triangularView<Eigen::Lower>().solve( DITdk );
					
					//            | L       0		|
					// Update L = | wT  sqrt(1-wTw)	|
					//                               
					L.conservativeResize( I_atom_count + 1, I_atom_count + 1 );
					L.row(I_atom_count).head(I_atom_count) = w.col(0).head(I_atom_count);
					L.col(I_atom_count).setZero(); 

					Scalar_t val_tmp = 1 - w.col(0).dot( w.col(0) );
					L( I_atom_count, I_atom_count ) = val_tmp < 1 ? (val_tmp < 0 ? 0 : (Scalar_t) sqrt( val_tmp )) : 1;
				}

				//std::cout << "Here is the matrix L:" << L << std::endl;

				PUSH_ARRAY_T( I_atoms, max_idx );
				I_atom_count++;

				DictI_T.conservativeResize( I_atom_count, dimensionality );
				DictI_T.row( I_atom_count - 1 ) = Dict.col( max_idx );

				//std::cout << "Here is the matrix DictI_T:" << DictI_T << std::endl;

				Matrix_t alpha_I( I_atom_count, 1 );
				alpha_I = DictI_T * ysample;
				// xI = solve for c { L.LT.c = alpha_I }
				// first solve LTc :
				Matrix_t LTc = L.triangularView<Eigen::Lower>().solve( alpha_I );
				// then solve xI :
				xI = L.transpose().triangularView<Eigen::Upper>().solve( LTc );

				// r = y - Dict_I * xI
				r = ysample - DictI_T.transpose() * xI;

				//std::cout << "Here is the new xI:" << xI << std::endl;
				//std::cout << "Here is the new residual:" << r << std::endl;
			}
		}

		// Update this particular sample in X matrix
		X.col( sample_idx ).setZero();
		for( int atom_idx = 0; atom_idx < I_atom_count; atom_idx++ )
		{
			X( I_atoms[atom_idx], sample_idx ) = xI( atom_idx, 0 );
		}
		//std::cout << "Here is the matrix X after updating sample " << sample_idx << std::endl << X << std::endl;

	}
}

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

void SolveImg( Scalar_t* img_data, int with, int height, Scalar_t* out_data )
{
	srand( 123 );

	const int block_size = 16; // 4x4
	int dictionary_size = (with * height / block_size) / 4;	// dictionary atoms picked at random first
	int sample_count = (with - 4) * (height - 4);

	Solver solver;
	solver.Init( 4 /*target_sparcity*/, dictionary_size /*dictionary_size*/, block_size /*dimensionality*/, sample_count /*sample_count*/ );

	// Fill Y matrix, which represents the original samples
	int sample_idx = 0;
	for( int y = 0; y < height - 4; y++)
	{
		for( int x = 0; x < with - 4; x++, sample_idx++)
		{
			for( int dimy = 0; dimy < 4; dimy++ )
			{
				for( int dimx = 0; dimx < 4; dimx++ )
				{
					int dim_index = dimy * 4 + dimx;
					solver.Y( dim_index, sample_idx ) = img_data[(y + dimy) * with + (x + dimx)];
				}
			}
		}
	}
	//std::cout << "Here is the matrix Y:" << std::endl << solver.Y << std::endl;

	// Initial dictionary
	for( int dict_idx = 0; dict_idx < dictionary_size ; dict_idx++ )
	{
		int rsample = rand() % sample_count;
		solver.Dict.col( dict_idx ) = solver.Y.col( rsample );
		solver.Dict.col( dict_idx ).normalize();
	}
	//std::cout << "Here is the matrix Dict:" << std::endl << solver.Dict << std::endl;

	const int max_iter = 20;
	for( int iter = 0; iter < max_iter ; iter++ )
	{
		solver.OMPStep();

		for( int kth = 0; kth < solver.dictionary_size ; kth++ )
		{
			std::cout << "ksvd step: " << kth << std::endl;
			solver.KSVDStep( kth );
		}
	}
}


}; /*namespace ksvd*/