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

void TestSolver()
{
	Solver solver;
	solver.Init( 1 /*target_sparcity*/, 4 /*dictionary_size*/, 2 /*dimensionality*/, 16 /*sample_count*/ );
	
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

	// Initial dictionnary
	const Scalar_t Sqrt2 = (Scalar_t)sqrt( 2.0 );
	//Matrix_t Dict( solver.dimensionality, solver.dictionary_size );
	solver.Dict(0, 0) = -Sqrt2;
	solver.Dict(1, 0) = -Sqrt2;
	solver.Dict(0, 1) = Sqrt2;
	solver.Dict(1, 1) = -Sqrt2;
	solver.Dict(0, 2) = -Sqrt2;
	solver.Dict(1, 2) = Sqrt2;
	solver.Dict(0, 3) = Sqrt2;
	solver.Dict(1, 3) = Sqrt2;

	// Init X
	for( int sample_idx = 0; sample_idx < 16; sample_idx++ )
	{
		for( int i=0; i<4; i++ )
			solver.X( i, sample_idx ) = (Scalar_t)((i == (sample_idx / 4)) ? 1 : 0);
	}

	// Encoded signal
	//Matrix_t X( solver.dictionary_size, solver.sample_count );

	for( int kth = 0; kth < solver.dictionary_size ; kth++ )
	{
		solver.KSVDStep( kth );
	}
}


}; /*namespace ksvd*/