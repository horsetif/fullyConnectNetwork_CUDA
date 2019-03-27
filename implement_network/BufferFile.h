#pragma once
#include<iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
using namespace std;
// Read file to buffer
class BufferFile {
public:
	std::string file_path_;
	std::size_t length_ = 0;
	std::unique_ptr<char[]> buffer_;

	explicit BufferFile(const std::string& file_path);

	std::size_t GetLength();

	char* GetBuffer();
};