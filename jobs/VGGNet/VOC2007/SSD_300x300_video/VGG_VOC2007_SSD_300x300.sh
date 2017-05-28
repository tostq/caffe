cd /home/lee/caffe
./build/tools/caffe test \
--model="models/VGGNet/VOC2007/SSD_300x300_video/test.prototxt" \
--weights="models/VGGNet/VOC2007/SSD_300x300/VGG_VOC2007_SSD_300x300_iter_60000.caffemodel" \
--iterations="536870911" \
--gpu 0
