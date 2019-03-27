#include "BufferFile.h"



BufferFile:: BufferFile(const std::string& file_path)
	: file_path_(file_path) {

	std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
	if (!ifs) {
		std::cerr << "Can't open the file. Please check . \n";
		return;
	}

	ifs.seekg(0, std::ios::end);
	length_ = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0, std::ios::beg);
	std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

	// Buffer as null terminated to be converted to string
	buffer_.reset(new char[length_ + 1]);
	buffer_[length_] = 0;
	ifs.read(buffer_.get(), length_);
	ifs.close();
}

std::size_t BufferFile::GetLength() {
	return length_;
}

char* BufferFile::GetBuffer() {
	return buffer_.get();
}
