#include <stdio.h>
#include "net.h"
#include "mnist.h"

//#define MTRACE

#ifdef MTRACE
#include <mcheck.h>
#endif

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


int main(void)
{
#ifdef MTRACE
	mtrace();
	printf("memory debug enabled.\n");
#endif
	struct net mynet;
	struct data_pack train_data, test_data;
	net_setup(&mynet, 3, neu_num);
	
	net_init(&mynet, sigmoid);
	
	read_data(&train_data, train_images_path, train_labels_path);
	read_data(&test_data, test_images_path, test_labels_path);
	
	//net_load(&mynet, filename);
	
	net_train(&mynet, &train_data, score, 20, 3.0, 1);
	
	float correct_rate = net_pack_test(&mynet, &test_data);
	printf("Correct rate : %.2f %% \n", correct_rate * 100);

	//net_work(&mynet, &train_data, 0, true);
	
	//mnist_print(&train_data, 0);
	
	//net_activ_save(&mynet, "activ");

	net_del(&mynet);
	mnist_del(&train_data);
	mnist_del(&test_data);

#ifdef MTRACE
	muntrace();
#endif
	return 0;
}

