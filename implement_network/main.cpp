#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include<stdio.h>
#include<iostream>
#include <windows.h>
#include"Network.h"
using namespace std;

int main()
{
	int N = 1000000; // 
	Network net("./weights/", N);
	for (int j = 0; j < 100000; j++) {
		float *test = (float*)malloc(sizeof(float)*N * 6);
		for (int i = 0; i < N * 6; i++) test[i] = 0.1;

		double start = GetTickCount();

		//float * result = net.run(test);
		float * result = net.run_network(test);

		double  end = GetTickCount();
		cout << "GetTickCount:  " << end - start << endl << endl << endl;
		//float GT[6] = { -0.10585209,0.06931,0.150405};
		//for (int i = 0; i < 3*N; i++) {
		//	cout << result[i] << "   " << GT[i/N] << endl;
		//}//
		delete[]test;
		delete[]result;
	}
	system("pause");
}