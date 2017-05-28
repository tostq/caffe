#include <caffe/caffe.hpp>
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/signal_handler.h"
#include "boost/algorithm/string.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>
#include <iostream>
//#include "SqueezeNet.hpp"
//#include "SSD.hpp"
#include "detect.hpp"
#include "convert_data.hpp"
using namespace cv;

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_string(mean_file, "",
	"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
	"If specified, can be one value or can be same as image channels"
	" - would subtract from the corresponding channel). Separated by ','."
	"Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
	"The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
	"If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.1,
	"Only store detections with score higher than the threshold.");

string inttostring(const unsigned int num){
	unsigned int w = 10;
	string str;
	while (num / w != 0){
		str = (char)(num%w + '0') + str;
		w = w * 10;
	}
	str = (char)(num / (w / 10) + '0') + str;
	return str;
}

void squeeze_test(){
	const string& model_file = "E:\\Code\\windows-ssd\\models\\SqueezeNet\\SqueezeNet_v1.0\\deploy.prototxt";
	const string& weights_file = "E:\\Code\\windows-ssd\\models\\SqueezeNet\\SqueezeNet_v1.0\\squeezenet_v1.0.caffemodel";

	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const string& file_type = FLAGS_file_type;
	const string& out_file = "E:\\image.txt";
	const float confidence_threshold = FLAGS_confidence_threshold;

	// Initialize the network.
	Detector detector(model_file, weights_file, mean_file, mean_value);
	if (file_type == "image") {
		unsigned int imagenum = 1;
		string outfileDir = "E:\\Code\\windows-ssd\\examples\\images\\airplane\\outfile\\";
		string fileDir = "E:\\Code\\windows-ssd\\examples\\images\\airplane\\";
		while (1){
			string file = fileDir + inttostring(imagenum) + ".jpg";
			string outfile = outfileDir + inttostring(imagenum) + ".jpg";
			cv::Mat img = cv::imread(file);
			CHECK(!img.empty()) << "Unable to decode image " << file;

			double t = (double)getTickCount();
			std::vector<vector<float> > detections = detector.Detect(img);
			t = ((double)getTickCount() - t) / getTickFrequency();
			std::cout << "image-" << imagenum << " detect time: " << t << std::endl;

			cv::imshow("abc", img);
			cv::imwrite(outfile, img);
			cv::waitKey(0);
			imagenum++;
		}
	}
}

void ssd_detect(){
	/* if (argc < 4) {
	gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
	return 1;
	}
	*/
	const string& model_file = "E:\\Code\\windows-ssd\\models\\SSD_300x300\\deploy.prototxt";
	const string& weights_file = "E:\\Code\\windows-ssd\\models\\SSD_300x300\\VGG_VOC0712_SSD_300x300_iter_60000.caffemodel";
	//const string& model_file = "E:\\Code\\windows-ssd\\models\\SqueezeNet\\SqueezeNet_v1.0\\deploy.prototxt";
	//const string& weights_file = "E:\\Code\\windows-ssd\\models\\SqueezeNet\\SqueezeNet_v1.0\\squeezenet_v1.0.caffemodel";

	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const string& file_type = FLAGS_file_type;
	const string& out_file = "E:\\image.txt";
	const float confidence_threshold = FLAGS_confidence_threshold;

	// Initialize the network.
	Detector detector(model_file, weights_file, mean_file, mean_value);

	// Set the output mode.
	std::streambuf* buf = std::cout.rdbuf();
	std::ofstream outfile;
	if (!out_file.empty()) {
		outfile.open(out_file.c_str());
		if (outfile.good()) {
			buf = outfile.rdbuf();
		}
	}
	std::ostream out(buf);

	// Process image one by one.

	// string file = "F:\\data\\原始数据\\行人数据及标签\\SSM\\s4.jpg";

	while (1)
	{
		if (file_type == "image") {
			unsigned int imagenum = 1;
			string outfileDir = "E:\\Code\\windows-ssd\\examples\\images\\airplane\\outfile\\";
			string fileDir = "E:\\Code\\windows-ssd\\examples\\images\\airplane\\";
			while (1){
				string file = fileDir + inttostring(imagenum) + ".jpg";
				string outfile = outfileDir + inttostring(imagenum) + ".jpg";
				cv::Mat img = cv::imread(file);
				//cv::Mat img = cv::imread(file);
				// IplImage* img = cvLoadImage(file, -1);
				// cvShowImage("1", img);

				//cv::imshow("abc", img);
				// cvWaitKey(0);
				CHECK(!img.empty()) << "Unable to decode image " << file;

				double t = (double)getTickCount();
				std::vector<vector<float> > detections = detector.Detect(img);
				t = ((double)getTickCount() - t) / getTickFrequency();
				std::cout << "image-" << imagenum << " detect time: " << t << std::endl;

				// cv::Mat img(imgs);
				/* Print the detection results. */
				for (int i = 0; i < detections.size(); ++i) {
					const vector<float>& d = detections[i];
					// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
					CHECK_EQ(d.size(), 7);
					const float score = d[2];
					if (score >= confidence_threshold) {


						// peng

						Point pt1, pt2;
						pt1.x = (img.cols*d[3]);
						pt1.y = (d[4] * img.rows);
						pt2.x = (d[5] * img.cols);
						pt2.y = (d[6] * img.rows);
						cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 1, 8, 0);
					}



				}
				cv::imshow("abc", img);
				cv::imwrite(outfile, img);
				cv::waitKey(0);
				imagenum++;
			}
		}

		else if (file_type == "video") {
			string file = "E:\\Code\\windows-ssd\\examples\\videos\\1.mp4";
			cv::VideoCapture cap(file);
			//Mat cap = imread("F:\\1.ts");
			if (!cap.isOpened()) {
				LOG(FATAL) << "Failed to open video: " << file;
			}
			cv::Mat img;
			int frame_count = 0;
			VideoCapture imgs("E:\\Code\\windows-ssd\\examples\\videos\\1.mp4");
			//VideoCapture imgs("E:\\数据\\test\\11\\1.mp4");
			while (true) {
				/*	bool success = cap.read(img);
				if (!success) {
				LOG(INFO) << "Process " << frame_count << " frames from " << file;
				break;
				}*/
				//CHECK(!img.empty()) << "Error when read frame";

				bool success = imgs.read(img);

				double t = (double)getTickCount();
				std::vector<vector<float> > detections = detector.Detect(img);
				t = ((double)getTickCount() - t) / getTickFrequency();
				std::cout << "frame-" << frame_count << " detect time: " << t << std::endl;



				/* Print the detection results. */
				for (int i = 0; i < detections.size(); ++i)
				{
					const vector<float>& d = detections[i];
					// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
					CHECK_EQ(d.size(), 7);
					const float score = d[2];
					if (score >= confidence_threshold) {
						/*	std::cout << file << "_";
						//std::cout << std::setfill('0') << std::setw(6) << frame_count << " ";
						std::cout << static_cast<int>(d[1]) << " ";
						std::cout << score << " ";
						std::cout << static_cast<int>(d[3] * img.cols) << " ";
						std::cout << static_cast<int>(d[4] * img.rows) << " ";
						std::cout << static_cast<int>(d[5] * img.cols) << " ";
						std::cout << static_cast<int>(d[6] * img.rows) << std::endl;*/
						Point pt1, pt2;
						pt1.x = (img.cols*d[3]);
						pt1.y = (d[4] * img.rows);
						pt2.x = (d[5] * img.cols);
						pt2.y = (d[6] * img.rows);
						cv::rectangle(img, pt1, pt2, cvScalar(0, 0, 255), 1, 8, 0);
					}

				}

				cv::imshow("abc", img);
				cv::waitKey(100);
				++frame_count;
				//			std::cout << frame_count << std::endl;

			}
			if (cap.isOpened()) {
				cap.release();
			}
		}
		else {
			LOG(FATAL) << "Unknown file_type: " << file_type;
		}
	}
}

// 分别从权重文件中复制权重到网络中
// 权重文件model_list输入形式：“file1,file2,file3,...”
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
	std::vector<std::string> model_names;
	boost::split(model_names, model_list, boost::is_any_of(",")); // 这里包括了多个权重文件
	for (int i = 0; i < model_names.size(); ++i) {
		LOG(INFO) << "Finetuning from " << model_names[i];
		solver->net()->CopyTrainedLayersFrom(model_names[i]); // 训练网络的权重文件
		for (int j = 0; j < solver->test_nets().size(); ++j) {
			solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]); // 测试网络的权重文件
		}
	}
}

// 网络训练
void train(const string& solverfile, const string& snapshotfile = string(""), const string& weightsfile = string("")) {
	CHECK_GT(solverfile.size(), 0) << "Need a solver definition to train.";
	CHECK(!snapshotfile.size() || !weightsfile.size())
		<< "Give a snapshot to resume training or weights to finetune "
		"but not both."; // 要么有训练快照，要么有训练权重

	// 读入训练超参数文件： 训练超参数文件除了提示网络描述文件外，还用于设置训练时的参数，比如学习速率、迭代次数等
	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie(solverfile, &solver_param);

	// 只使用CPU运行
	LOG(INFO) << "Use CPU.";
	Caffe::set_mode(Caffe::CPU);

	// 设置训练中断后操作
	// 这里前者表示SIGINT信号（即由Ctrl+c产生）停止训练STOP
	// 后者表示SIGHUP信号（即表示用户连接结束）保存快照SNAPSHOT
	// 什么都不操作NONE
	caffe::SignalHandler signal_handler(caffe::SolverAction::STOP, caffe::SolverAction::SNAPSHOT);
	shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	solver->SetActionFunction(signal_handler.GetActionFunction());

	if (snapshotfile.size()) {
		LOG(INFO) << "Resuming from " << snapshotfile; //从快照中恢复
		solver->Restore(snapshotfile.c_str());
	}
	else if (weightsfile.size()) {  // 给定权重进行初始化
		CopyLayers(solver.get(), weightsfile);
	}

	LOG(INFO) << "Starting Optimization";
	solver->Solve();
	LOG(INFO) << "Optimization Done.";
}

// 提取测试中的出错样本
void extractWrongTest(const string& modelfile, const string& weightsfile, const int iterations, const char* resultfile){
	// 检查是否存在模型文件和权重文件
	CHECK_GT(modelfile.size(), 0) << "Need a model definition to score.";
	CHECK_GT(weightsfile.size(), 0) << "Need model weights to score.";

	// 只使用CPU
	LOG(INFO) << "Use CPU.";
	Caffe::set_mode(Caffe::CPU);

	// 建立模型网络，并导入权重.
	Net<float> caffe_net(modelfile, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(weightsfile);
	LOG(INFO) << "Running for " << weightsfile << " iterations.";

	// 出错样本的保存文件
	//std::ofstream wrongtest(resultfile, std::ios::out | std::ios::binary);
	FILE  *wrongtest = NULL;
	wrongtest = fopen(resultfile, "wb");
	CHECK(wrongtest) << "Unable to open file " << resultfile;

	vector<int> test_score_output_id;
	vector<float> test_score; // 测试得分
	float loss = 0;
	for (int i = 0; i < iterations; ++i) {
		// 这里result保存网络输出，iter_loss表示损失
		float iter_loss;
		const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);
		const vector<string>& blobnames = caffe_net.blob_names();
		const vector<string>& layernames = caffe_net.layer_names();
		const shared_ptr<Blob<float>> accblob = caffe_net.blob_by_name("ip2_ip2_0_split_0");
		const shared_ptr<Blob<float>> labelblob = caffe_net.blob_by_name("label");
		const shared_ptr<Blob<float>> datalob = caffe_net.blob_by_name("data");

		const int outer_num = labelblob->num();
		const int inner_num = labelblob->channels();
		const int top_k = 1;

		float* input_data = datalob->mutable_cpu_data();
		const float* bottom_data = accblob->cpu_data();
		const float* bottom_label = labelblob->cpu_data();
		const int dim = accblob->count() / outer_num;
		const int num_labels = accblob->channels(); // 标签数


		int count = 0;
		for (int i = 0; i < outer_num; ++i) {
			for (int j = 0; j < inner_num; ++j) {
				const int label_value = static_cast<int>(bottom_label[i * inner_num + j]);

				DCHECK_GE(label_value, 0);
				DCHECK_LT(label_value, num_labels);
				// Top-k accuracy
				std::vector<std::pair<float, int> > bottom_data_vector;
				for (int k = 0; k < num_labels; ++k) {
					bottom_data_vector.push_back(std::make_pair(
						bottom_data[i * dim + k * inner_num + j], k));
				}
				std::partial_sort(
					bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
					bottom_data_vector.end(), std::greater<std::pair<float, int> >());
				// check if true label is in top k predictions
				for (int k = 0; k < top_k; k++) {
					if (bottom_data_vector[k].second == label_value) {
						break;
					}
					if (k == top_k - 1){ //没有匹配的情况
						fwrite(&i, sizeof(int), 1, wrongtest);
						fwrite(&bottom_data_vector[k].second, sizeof(int), 1, wrongtest);
						fwrite(&label_value, sizeof(int), 1, wrongtest);
						float* imgdata = input_data + i * 28 * 28;
						fwrite(imgdata, sizeof(float), 28 * 28, wrongtest);
						/*
						cv::Mat img(28, 28, CV_32FC1, imgdata);
						cv::imshow("abc", img);
						cv::waitKey(0);
						*/
					}
				}
			}
		}
	}
	fclose(wrongtest);
}

// 测试模型，modelfile模型文件，weightsfile权重文件，iterations测试迭代次数
void test(const string& modelfile, const string& weightsfile, const int iterations){
	// 检查是否存在模型文件和权重文件
	CHECK_GT(modelfile.size(), 0) << "Need a model definition to score.";
	CHECK_GT(weightsfile.size(), 0) << "Need model weights to score.";

	// 只使用CPU
	LOG(INFO) << "Use CPU.";
	Caffe::set_mode(Caffe::CPU);

	// 建立模型网络，并导入权重.
	Net<float> caffe_net(modelfile, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(weightsfile);
	LOG(INFO) << "Running for " << weightsfile << " iterations.";

	vector<int> test_score_output_id;
	vector<float> test_score; // 测试得分
	float loss = 0;
	for (int i = 0; i < iterations; ++i) {
		// 这里result保存网络输出，iter_loss表示损失
		float iter_loss;
		const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);

		loss += iter_loss;
		int idx = 0;
		for (int j = 0; j < result.size(); ++j) {
			const float* result_vec = result[j]->cpu_data();
			for (int k = 0; k < result[j]->count(); ++k, ++idx) {
				const float score = result_vec[k];
				if (i == 0) {
					test_score.push_back(score);
					test_score_output_id.push_back(j);
				}
				else {
					test_score[idx] += score;
				}
				const std::string& output_name = caffe_net.blob_names()[
					caffe_net.output_blob_indices()[j]];
				LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
			}
		}
	}
	
	loss /= iterations;
	LOG(INFO) << "Loss: " << loss;
	for (int i = 0; i < test_score.size(); ++i) {
		const std::string& output_name = caffe_net.blob_names()[
			caffe_net.output_blob_indices()[test_score_output_id[i]]];
		const float loss_weight = caffe_net.blob_loss_weights()[
			caffe_net.output_blob_indices()[test_score_output_id[i]]];
		std::ostringstream loss_msg_stream;
		const float mean_score = test_score[i] / iterations;
		if (loss_weight) {
			loss_msg_stream << " (* " << loss_weight
				<< " = " << loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
	}
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif
	gflags::SetUsageMessage("Do detection using SSD mode.\n"
		"Usage:\n"
		"    ssd_detect [FLAGS] model_file weights_file list_file\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	
	//ssd_detect();
	train("E:/Code/windows-ssd/models/SSD_300x300/solver.prototxt");
	//train("./cifar10/cifar10_quick_solver.prototxt");
	//extractWrongTest("./mnist/lenet_train_test.prototxt", "./mnist/lenet_iter_5000.caffemodel", 100,"./mnist/wrongtest.dat");
	//test("./mnist/lenet_train_test.prototxt", "./mnist/lenet_iter_5000.caffemodel", 100);
	//convert_mnist_dataset("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte",	"./mnist/mnist_train_lmdb");
	//convert_imgs_dataset("./mnist/filenames.txt", "./mnist/labelfile.txt", "./mnist/test_lmdb");

#else
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV

	return 0;
}