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

// ����ת�������ݼ���ʽ��Ĭ��Ϊlmdb��Ҳ����Ϊleveldb
DEFINE_string(backend, "lmdb", "The backend for storing the result");

// ��С�˵�ת��
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void convert_mnist_dataset(const char* image_filename, const char* label_filename,
        const char* db_path, const string& db_backend) {
	// ���ļ�
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// Read the magic and the meta data
	uint32_t magic;       // ���ݱ�ʶ��2051ָѵ��ͼ��2049ָѵ����ǩ
	uint32_t num_items;   // ͼ����Ŀ����
	uint32_t num_labels;  // ��ǩ��Ŀ����
	uint32_t rows;        // ͼ������
	uint32_t cols;        // ͼ������

	// ��ͼ������
	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);
	// ����ǩ����
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);

	//scoped_ptrָ����std::auto_ptr���ƣ������������䲻��ת������Ȩ
	scoped_ptr<db::DB> db(db::GetDB(db_backend)); // ����õ�һ��lmdb��ʽ��DB
	db->Open(db_path, db::NEW); // �򿪲�����DB
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
	// ���������д��
	if (count % 1000 != 0) {
		txn->Commit();
	}
	LOG(INFO) << "Processed " << count << " files.";
	delete[] pixels;
	db->Close();
}

void convert_imgs_dataset(const char* image_filename, const char* label_filename,
	const char* db_path, const string& db_backend){
	// ����image_filename���ͼ���ļ���ַ��label_filename��ű�ǩ�ļ���ַ��db_path��Ž����ַ
	// ���ļ�
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;


	//scoped_ptrָ����std::auto_ptr���ƣ������������䲻��ת������Ȩ	
	scoped_ptr<db::DB> db(db::GetDB(db_backend)); // ����õ�һ��lmdb��ʽ��DB	
	db->Open(db_path, db::NEW); // �򿪲�����DB
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// ����ͼ���ַ�ͱ�ǩ
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

	// �����Ƿ���Ҫת��ͼ���С������Ϊ0��ʾ����Ҫ���д�Сת��
	int resize_height = 0;
	int resize_width = 0;
	const bool is_color = true; // ��ʾ��ɫͼ��������лҶ�ת��
	const bool check_size = false; // �Ƿ���Ҫ��飺����ͼ��Datum������ͬ��С
	const string encode = "";

	for (int image_id = 0; image_id < images.size(); ++image_id) {
		// �����ݶ���Datum
		bool status;
		status = ReadImageToDatum(images[image_id].first,
			images[image_id].second, resize_height, resize_width, is_color,
			encode, &datum);
		if (status == false) continue;
		// ���ÿ�ζ����ͼ���С�Ƿ�һ��
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
