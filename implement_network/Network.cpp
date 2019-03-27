#include "Network.h"
#include <cuda_runtime.h>

//the CUDA kernel function of matrix broadcast and relu
__global__ void broadcast_relu(float* C,float *bias, const int fixed_num , const int N) {
	const unsigned int thread_idx = blockIdx.x + threadIdx.y * N;
	C[thread_idx] = C[thread_idx] + bias[threadIdx.y];
	if (C[thread_idx] < fixed_num) C[thread_idx] = 0;
};

//the CUDA kernel function of matrix broadcast only
__global__ void broadcast(float* C, float *bias, const int N) {
	const unsigned int thread_idx = blockIdx.x + threadIdx.y * N;
	C[thread_idx] = C[thread_idx] + bias[threadIdx.y];
};


Network::Network(const std::string& path_name,int total_num)
	:_path_name(path_name),_total_num(total_num)
{
	read_txt(path_name + "fullyconnected0_weight.txt", weight0);
	read_txt(path_name + "fullyconnected1_weight.txt", weight1);
	read_txt(path_name + "fullyconnected2_weight.txt", weight2);
	read_txt(path_name + "fullyconnected3_weight.txt", weight3);
	read_txt(path_name + "fullyconnected0_bias.txt", bias0);
	read_txt(path_name + "fullyconnected1_bias.txt", bias1);
	read_txt(path_name + "fullyconnected2_bias.txt", bias2);
	read_txt(path_name + "fullyconnected3_bias.txt", bias3);

	//read the data and exchange the data in GPU memory.
	cudaMalloc((void **)&d_weight0, sizeof(float)* 64 * 6);
	cudaMemcpy(d_weight0, weight0, sizeof(float) * 64 * 6, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_weight1, sizeof(float) * 128 * 64);
	cudaMemcpy(d_weight1, weight1, sizeof(float) * 128 * 64, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_weight2, sizeof(float) * 64 * 128);
	cudaMemcpy(d_weight2, weight2, sizeof(float) * 64 * 128, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_weight3, sizeof(float) * 64 * 3);
	cudaMemcpy(d_weight3, weight3, sizeof(float) * 64 * 3, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_bias0, sizeof(float) * 64);
	cudaMemcpy(d_bias0, bias0, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_bias1, sizeof(float) * 128);
	cudaMemcpy(d_bias1, bias1, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_bias2, sizeof(float) * 64);
	cudaMemcpy(d_bias2, bias2, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_bias3, sizeof(float) * 3);
	cudaMemcpy(d_bias3, bias3, sizeof(float) * 3, cudaMemcpyHostToDevice);
	
}


Network::~Network()
{
	//delete[] weight0;
	//delete[] bias0;
	//delete[] weight1;
	//delete[] bias1;
	//delete[] weight2;
	//delete[] bias2;
	//delete[] weight3;
	//delete[] bias3;
	//release the GPU memory.
	cudaFree(d_weight0);
	cudaFree(d_weight1);
	cudaFree(d_weight2);
	cudaFree(d_weight3);
	cudaFree(d_bias0);
	cudaFree(d_bias1);
	cudaFree(d_bias2);
	cudaFree(d_bias3);
}

//void Network::read_txt(string path, float* p, int M, int N, bool is_bias) {
//	//p = (float*)malloc(sizeof(float)*M*N);
//	//p = new float[M*N];
//	ifstream infile;
//	infile.open(path, ios::in);
//	if (!infile.is_open())
//		cout << "open file failure!" << endl;
//	int index = 0;
//	while (!infile.eof()) {
//		infile >> p[index++];
//		if (is_bias) {
//			float temp = p[index - 1];
//			for (int i = 0; i < N - 1; i++) {
//				p[index++] = temp;
//			}
//		}
//	}
//	infile.close();
//	//if (is_bias) {
//	//	for (int i = 0; i < M*N; i++) {
//	//		cout << p[i] << " ";
//	//		if ((i+1)%N == 0) cout << endl;
//	//	}
//	//	system("pause");
//	//}
//
//}

void Network::read_txt(string path, float* p) {

	ifstream infile;
	infile.open(path, ios::in);
	if (!infile.is_open())
		cout << "open file failure!" << endl;
	int index = 0;
	while (!infile.eof()) {
		infile >> p[index++];
	}
	infile.close();

}
//input and output are CPU memory,multiply the matrix using CUDA. 
float* Network::matrixMultiply(float * A, float * B, float * C, int m, int n, int k)
{
	float *d_A, *d_B, *d_C, *d_C_bias;
	unsigned int mem_size_A = sizeof(float) * m * k;
	unsigned int mem_size_B = sizeof(float) * k * n;
	unsigned int mem_size_C = sizeof(float) * m * n;

	float *h_CUBLAS = (float *)malloc(mem_size_C);
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);
	cudaMalloc((void**)&d_C_bias, sizeof(float)*m);
	cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0, mem_size_C);
	//cudaMemcpy(d_C, C, mem_size_C, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_bias, C, sizeof(float)*m, cudaMemcpyHostToDevice);

	dim3 threads(1, 1);
	dim3 grid(1, 1);

	//cuBLAS代码
	const float alpha = 1.0f;
	const float beta = 1.0f;
	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

	cublasDestroy(handle);

	dim3 dimBlock(1,m, 1);
	dim3 dimGrid(n, 1, 1);
	broadcast <<<dimGrid, dimBlock >>> (d_C,d_C_bias,n);
	cudaThreadSynchronize();

	cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return h_CUBLAS;
}
//input and output are CPU memory,multiply and relu the matrix using CUDA. 
float * Network::matrixMultiply_relu(float * A, float * B, float * C, int m, int n, int k)
{
	float *d_A, *d_B, *d_C, *d_C_bias;
	unsigned int mem_size_A = sizeof(float) * m * k;
	unsigned int mem_size_B = sizeof(float) * k * n;
	unsigned int mem_size_C = sizeof(float) * m * n;

	float *h_CUBLAS = (float *)malloc(mem_size_C);
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);
	cudaMalloc((void**)&d_C_bias, sizeof(float)*m);
	cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0, mem_size_C);
	//cudaMemcpy(d_C, C, mem_size_C, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_bias, C, sizeof(float)*m, cudaMemcpyHostToDevice);

	dim3 threads(1, 1);
	dim3 grid(1, 1);

	//cuBLAS代码
	const float alpha = 1.0f;
	const float beta = 1.0f;
	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
	 
	cublasDestroy(handle);

	dim3 dimBlock(1, m, 1);
	dim3 dimGrid(n, 1, 1);
	broadcast_relu <<<dimGrid, dimBlock >>> (d_C, d_C_bias, 0, n);
	cudaThreadSynchronize();

	cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return h_CUBLAS;
}
//input and output are GPU memory,multiply the matrix using CUDA. 
float* Network::matrixMultiply_GPU(float *d_A, float *d_B, int m, int n, int k) {
	float *d_C;
	cudaMalloc((void**)&d_C, sizeof(float)*m*n);
	cudaMemset(d_C, 0, sizeof(float)*m*n);
	//cuBLAS代码
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

	cublasDestroy(handle);
	return d_C;
}
//input and output are GPU memory,multiply and relu the matrix using CUDA. 
float* Network::broadcast_relu_GPU(float *A, float *d_bias, int m, int n, int fix_num) {
	dim3 dimBlock(1, m, 1);
	dim3 dimGrid(n, 1, 1);
	broadcast_relu << <dimGrid, dimBlock >> > (A, d_bias, 0, n);
	cudaThreadSynchronize();
	return A;
}
float* Network::broadcast_GPU(float *A, float *d_bias, int m, int n) {
	dim3 dimBlock(1, m, 1);
	dim3 dimGrid(n, 1, 1);
	broadcast << <dimGrid, dimBlock >> > (A, d_bias, n);
	cudaThreadSynchronize();
	return A;
}
//a network that each layer using GPU memory exchange.
float *Network::run_network(float *input) {
	float *d_input;
	cudaMalloc((void**)&d_input, sizeof(float) * 6 * _total_num);
	cudaMemcpy(d_input, input, sizeof(float) * 6 * _total_num, cudaMemcpyHostToDevice);

	double start = GetTickCount();
	float *layer_0 = matrixMultiply_GPU(d_weight0, d_input, 64, _total_num, 6);
	float *layer_0_relu = broadcast_relu_GPU(layer_0,d_bias0, 64, _total_num, 0);

	float *layer_1 = matrixMultiply_GPU(d_weight1, layer_0_relu, 128, _total_num, 64);
	float *layer_1_relu = broadcast_relu_GPU(layer_1, d_bias1, 128, _total_num, 0);

	float *layer_2 = matrixMultiply_GPU(d_weight2, layer_1_relu, 64, _total_num, 128);
	float *layer_2_relu = broadcast_relu_GPU(layer_2, d_bias2, 64, _total_num, 0);

	float *layer_3 = matrixMultiply_GPU(d_weight3, layer_2_relu, 3, _total_num, 64);
	float *layer_3_bd = broadcast_GPU(layer_3, d_bias3, 3, _total_num);

	double  end = GetTickCount();
	cout << "GetTickCount only calculate:  " << end - start << endl << endl << endl;

	float *layer_3_CPU = (float *)malloc(sizeof(float) * 3 * _total_num);
	cudaMemcpy(layer_3_CPU, layer_3_bd, sizeof(float) * 3 * _total_num, cudaMemcpyDeviceToHost);

	cudaFree(layer_0_relu);
	cudaFree(layer_1_relu);
	cudaFree(layer_2_relu);
	cudaFree(layer_3_bd);

	return layer_3_CPU;
}

//a network that each layer using CPU memory exchange.
float * Network::run(float * input)
{
	//if (N != 1) return NULL;
	//for (int i = 0; i < 64; i++) cout << bias0[i] << endl;
	float *layer1= matrixMultiply_relu(weight0, input, bias0, 64, _total_num, 6);
	float *layer2 = matrixMultiply_relu(weight1, layer1, bias1, 128, _total_num, 64);
	float *layer3 = matrixMultiply_relu(weight2, layer2, bias2, 64, _total_num, 128);
	float *layer4 = matrixMultiply(weight3, layer3, bias3, 3,_total_num, 64);
	return layer4;
}


// there are some problems in this function.
void Network::getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}