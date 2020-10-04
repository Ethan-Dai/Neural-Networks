#ifndef MNIST_H
#define MNIST_H

struct img {
	unsigned char label;
	unsigned char data[784];
};

struct data_pack {
	unsigned int img_num;
	unsigned int img_size;
	struct img *imgs;
};

int read_data(struct data_pack *pack, const char *img_path, const char *label_path);

void mnist_del(struct data_pack *data_pack);

void mnist_print(struct data_pack *pack, unsigned int index);

int net_load_mnist(struct data_pack *pack, unsigned int index, double *net_input, unsigned int input_size);
#endif /* MNIST_H */

