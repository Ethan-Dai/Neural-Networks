#include <stdio.h>
#include "net.h"
#include "mnist.h"

const unsigned int neu_num[3]={784, 30, 10};

const double score[10*10] = {
1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0
0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 1
0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 2
0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 3
0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 4
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // 5
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // 6
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // 7
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // 8
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0  // 9
};

const char *train_images_path = "../Data/train-images";
const char *train_labels_path = "../Data/train-labels";

const char *test_images_path = "../Data/t10k-images";
const char *test_labels_path = "../Data/t10k-labels";

char filename[20];

int main(void)
{
	struct net mynet;
	struct data_pack train_data, test_data;
	net_setup(&mynet, 3, neu_num);

	net_init(&mynet, sigmoid);
	//net_load(&mynet, "net80");
	
	read_data(&train_data, train_images_path, train_labels_path);
	read_data(&test_data, test_images_path, test_labels_path);
	
	net_train(&mynet, &train_data, score, 5, 0.2, 1);

	/* test */

	//net_work(&mynet, &train_data, 0, true);
	
	float correct_rate = net_pack_test(&mynet, &test_data);
	printf("Correct rate : %.2f %% \n", correct_rate * 100);

	sprintf(filename, "net%d", (int)(100 * correct_rate));
	net_save(&mynet, filename);
	
	//mnist_print(&train_data, 0);
	
	//net_activ_save(&mynet, "activ");

	net_del(&mynet);
	mnist_del(&train_data);
	return 0;
}

