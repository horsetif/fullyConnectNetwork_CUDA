#pragma once
#include<iostream>
#include<vector>
#include<string>
#include <io.h>
#include<fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include<device_launch_parameters.h>
#include <windows.h>
using namespace std;
class Network
{
private:
	string _path_name;
	vector<string> file_name;
	float weight0[64*6];
	float bias0[64];
	float weight1[128*64];
	float bias1[128];
	float weight2[64*128];
	float bias2[64];
	float weight3[64*3];
	float bias3[3];
	float* d_weight0, *d_weight1, *d_weight2, *d_weight3;
	float* d_bias0, *d_bias1, *d_bias2, *d_bias3;
	int _total_num;
public:
	Network(const std::string& path_name,int total_num);
	void getFiles(string path, vector<string>& files);
	void read_txt(string path,float *p);
	float* matrixMultiply(float *A, float *B, float *C, int m, int n, int k);
	float* matrixMultiply_relu(float *A, float *B, float *C, int m, int n, int k);
	float* matrixMultiply_GPU(float *A, float *B, int m, int n, int k);
	float* broadcast_relu_GPU(float *A, float* bias,int m, int n, int fix_num);
	float* broadcast_GPU(float *A, float* bias, int m, int n);
	float* run(float *input);
	float* run_network(float *input);
	
	~Network();
};

