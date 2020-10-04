#include "mnist.h"
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define r32b(p)  (*(p) << 24 | *(p + 1) << 16 | *(p + 2) << 8 | *(p + 3))

int read_chars(unsigned char *buff, int len, FILE *file)
{
	for (int i = 0; i < len; i++) {
		buff[i] = fgetc(file);
	}
	return 0;
}


int read_data(struct data_pack *pack, const char *img_path, const char *label_path)
{
	unsigned char img_head[16];
	unsigned char label_head[8];

	FILE *mnist_img = fopen(img_path, "r");
	if (mnist_img == NULL) {
		printf("Can't open file: %s.\n", img_path);
		exit(-ENOENT);
	}

	FILE *mnist_label = fopen(label_path, "r");
	if (mnist_label == NULL) {
		printf("Can't open file: %s.\n", label_path);
		exit(-ENOENT);
	}

	read_chars(img_head, 16, mnist_img);
	read_chars(label_head, 8, mnist_label);
	
	/* Check whether the file is MNIST data */
	if (r32b(img_head) != 2051) {
		printf("file: %s is not MNIST image data!\n", img_path);
		printf("magic: %d\n", r32b(img_head));
		exit(-EINVAL);
	}
	if (r32b(label_head) != 2049) {
		printf("file: %s is not MNIST label data!\n", label_path);
		printf("magic: %d\n", r32b(label_head));
		exit(-EINVAL);
	}
	
	/* number of images and labels */
	if (r32b(img_head + 4) != r32b(label_head + 4)) {
		printf("file: %s and file: %s nort match!\n", img_path, label_path);
		exit(-EINVAL);
	} else {
		pack->img_num = r32b(img_head + 4);
	}
	
	pack->img_size = 784;

	pack->imgs = (struct img *)malloc(pack->img_num * sizeof(struct img));
	if (pack->imgs == NULL)
		exit(-ENOMEM);
	
	printf("Start reading images and labels...   ");
	for (int i = 0; i < pack->img_num; i++) {
		struct img *img_i = pack->imgs + i;
		img_i->label = fgetc(mnist_label);
		read_chars(img_i->data, 784, mnist_img);
	}
	printf("Done!\n");

	fclose(mnist_img);
	fclose(mnist_label);
	return 0;
}

void mnist_del(struct data_pack *data_pack)
{
	free(data_pack->imgs);
}


void mnist_print(struct data_pack *pack, unsigned int index)
{
	struct img *img = pack->imgs + index;
	printf("label: %d", img->label);
	for (int i = 0; i < 784; i ++){
		printf("%c", img->data[i] > 128? '*' : ' ');
		if (i % 28 == 27)
			printf("\n");
	}
}


int net_load_mnist(struct data_pack *pack, unsigned int index,
			double *net_input, unsigned int input_size)
{
	if (input_size != pack->img_size) {
		printf("Load image to net failed! ");
		printf("Image Size : %d, Net Input: %d.\n",
			pack->img_size, input_size);
		printf("%s @ %s:%d.\n", strerror(EINVAL), __FILE__, __LINE__);
		exit(EINVAL);
	}
	struct img *img = pack->imgs + index;
	for (int i = 0; i < 784; i ++){
		net_input[i] = img->data[i] / 256.0;
	}
	return (int)img->label;
}
