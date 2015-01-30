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
	for( int sample_idx = 0; sample_idx < sample_count; sample_idx++)
	{
		if( X(kth, sample_idx) != 0 )
		{
			PUSH_ARRAY_T( wk, sample_idx );
		}
	}
}

void TestSolver()
{
	Solver solver;
	solver.Init( 1 /*target_sparcity*/, 4 /*dictionary_size*/, 2 /*dimensionality*/, 16 /*sample_count*/ );
	
	// Fill Y matrix, which represents the original samples
	//Matrix_t Y( solver.dimensionality, solver.sample_count );
	for( int group_idx = 0; group_idx < 4 ; group_idx++ )
	{
		Scalar_t group_x = (Scalar_t)-0.5 + (float)(group_idx % 2);
		Scalar_t group_y = (Scalar_t)-0.5 + (float)(group_idx / 2);

		for( int sub_group_idx = 0; sub_group_idx < 4 ; sub_group_idx++ )
		{
			Scalar_t sub_group_x = group_x - (Scalar_t)0.1 + (Scalar_t)(sub_group_idx % 2);
			Scalar_t sub_group_y = group_y - (Scalar_t)0.1 + (Scalar_t)(sub_group_idx / 2);

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

	// Encoded signal
	//Matrix_t X( solver.dictionary_size, solver.sample_count );

	for( int kth = 0; kth < solver.dictionary_size ; kth++ )
	{
		solver.KSVDStep( kth );
	}
}


}; /*namespace ksvd*/