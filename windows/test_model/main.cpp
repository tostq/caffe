#include<iostream>
#include<vector>
#include<caffe/blob.hpp>
using namespace caffe;
using namespace std;

int main()
{
	Blob<float> a;
	cout << "size:" << a.shape_string() << endl;
	return 0;
}