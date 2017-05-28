// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/
#include "../convert_data.hpp"

#if defined(USE_LEVELDB) && defined(USE_LMDB)

using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;
using std::string;
using namespace cv;

// 定义转换的数据集格式，默认为lmdb，也可以为leveldb
DEFINE_string(backend, "lmdb", "The backend for storing the result");

// 大小端的转换
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void convert_mnist_dataset(const char* image_filename, const char* label_filename,
        const char* db_path, const string& db_backend) {
	// 打开文件
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// Read the magic and the meta data
	uint32_t magic;       // 数据标识，2051指训练图像，2049指训练标签
	uint32_t num_items;   // 图像条目总数
	uint32_t num_labels;  // 标签条目总数
	uint32_t rows;        // 图像行数
	uint32_t cols;        // 图像列数

	// 读图像数据
	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);
	// 读标签数据
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);

	//scoped_ptr指针与std::auto_ptr类似，而区别在于其不能转让所有权
	scoped_ptr<db::DB> db(db::GetDB(db_backend)); // 这里得到一个lmdb格式的DB
	db->Open(db_path, db::NEW); // 打开并创建DB
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	char label;
	char* pixels = new char[rows * cols];
	int count = 0;
	string value;

	Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
	for (int item_id = 0; item_id < num_items; ++item_id) {
		image_file.read(pixels, rows * cols);
		label_file.read(&label, 1);
		datum.set_data(pixels, rows*cols);
		datum.set_label(label);
		string key_str = caffe::format_int(item_id, 8);
		datum.SerializeToString(&value);

		txn->Put(key_str, value);

		if (++count % 1000 == 0) {
			txn->Commit();
		}
	}
	// 把最后数据写入
	if (count % 1000 != 0) {
		txn->Commit();
	}
	LOG(INFO) << "Processed " << count << " files.";
	delete[] pixels;
	db->Close();
}

void convert_imgs_dataset(const char* image_filename, const char* label_filename,
	const char* db_path, const string& db_backend){
	// 这里image_filename存放图像文件地址，label_filename存放标签文件地址，db_path存放结果地址
	// 打开文件
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;


	//scoped_ptr指针与std::auto_ptr类似，而区别在于其不能转让所有权	
	scoped_ptr<db::DB> db(db::GetDB(db_backend)); // 这里得到一个lmdb格式的DB	
	db->Open(db_path, db::NEW); // 打开并创建DB
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// 读入图像地址和标签
	std::vector<std::pair<std::string, int> > images;
	string imagename;
	int label;
	while (image_file >> imagename){
		label_file >> label;
		images.push_back(std::make_pair(imagename, label));
	}
	label_file.close();
	image_file.close();

	// Storing to db
	Datum datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	// 定义是否需要转换图像大小，设置为0表示不需要进行大小转换
	int resize_height = 0;
	int resize_width = 0;
	const bool is_color = true; // 表示彩色图像，无需进行灰度转换
	const bool check_size = false; // 是否需要检查：所有图像Datum具有相同大小
	const string encode = "";

	for (int image_id = 0; image_id < images.size(); ++image_id) {
		// 将数据读入Datum
		bool status;
		status = ReadImageToDatum(images[image_id].first,
			images[image_id].second, resize_height, resize_width, is_color,
			encode, &datum);
		if (status == false) continue;
		// 检测每次读入的图像大小是否一致
		if (check_size) {
			if (!data_size_initialized) {
				data_size = datum.channels() * datum.height() * datum.width();
				data_size_initialized = true;
			}
			else {
				const std::string& data = datum.data();
				CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
					<< data.size();
			}
		}
		// sequential
		string key_str = caffe::format_int(image_id, 8) + "_" + images[image_id].first;

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
}

#endif  // USE_LEVELDB and USE_LMDB
