#include "net.h"
#include <malloc.h>
#include <errno.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EXIT(e) { \
		printf("ERROR: %s @%s:%d\n", strerror(e), __FILE__, __LINE__);\
		exit(e); }

void array_sub(double *a1, double *a2, double *output, unsigned int len)
{
	for (int i = 0; i < len; i++)
		output[i] = a1[i] - a2[i];
}

void array_wsub(double *a1, double *a2, double *output,
		double weight, unsigned int len)
{
	for (int i = 0; i < len; i++)
		output[i] = a1[i] - a2[i] * weight;
}

double guass_rand()
{
	int sum = 0;
	for (int i = 0; i < 4 ; i++){
		srand(clock());
		sum += (rand() - RAND_MAX / 2) % 1024;
	}
	return sum/4096.0;
}


int net_setup(struct net *net, const unsigned int layer_num,
					const unsigned int *neurous_num)
{
	net->layer_num = layer_num;

	/* alloc memory for layer structs */
	net->layers = 
		(struct layer *)malloc(layer_num * sizeof(struct layer));
	if (net->layers == NULL)
		EXIT(ENOMEM);
	
	/* alloc memory for link structs */
	net->links = 
		(struct link *)malloc((layer_num - 1) * sizeof(struct link));
	if (net->links == NULL)
		EXIT(ENOMEM);

	/* alloc memory for nt, activ and biases */
	for (int i = 0; i < layer_num; i++) {

		struct layer *layer_i = net->layers + i;

		layer_i->neurous_num = neurous_num[i];
		layer_i->activ = 
				(double *)malloc(neurous_num[i]*sizeof(double));
		if (layer_i -> activ == NULL)
			EXIT(ENOMEM);

		if (i == 0)   // biases and nt in first layer is useless.
			continue;

		layer_i->nt = (double *)malloc(neurous_num[i]*sizeof(double));
		layer_i->biases = 
				(double *)malloc(neurous_num[i]*sizeof(double));
		if (layer_i->nt == NULL || layer_i->biases == NULL)
			EXIT(ENOMEM);
	}

	/* alloc memory for link's weights */
	for (int i = 0; i < layer_num - 1; i++) {
		struct link *link_i = net->links + i;
		link_i->prev = net->layers + i;
		link_i->next = net->layers + i + 1;
		link_i->link_num = neurous_num[i] * neurous_num[i+1];
		link_i->weights = 
			(double *)malloc(link_i->link_num * sizeof(double));
		if (link_i->weights == NULL)
			EXIT(ENOMEM);
	}
	
	
	return 0;
}

void net_del(struct net *net)
{
	unsigned int layer_num = net->layer_num;

	for (int i = 0; i < layer_num; i++) {
		struct layer *layer_i = net->layers + i;
		free(layer_i->nt);
		free(layer_i->activ);
		free(layer_i->biases);
	}
	
	for (int i = 0; i < layer_num - 1; i++) {
		struct link *link_i = net->links + i;
		free(link_i->weights);
	}

	free(net->layers);
	free(net->links);
}

void net_print(struct net *net)
{
	printf("layer num: %d \n", net->layer_num);
	printf("activation mode: %d\n\n", net->activ_mode);

	/* print the biases value layer by layer */
	for (int i = 0; i < net->layer_num; i++) {
		struct layer *layer_i = net->layers + i;
		printf("layer%d : %d biases.\n", i, layer_i->neurous_num);
		if (i == 0) {
			printf("biases in first layer is useless.");
			printf("\nlayer%d end.\n\n", i);
			continue;
		}
		for(int j = 0; j < layer_i->neurous_num; j++)
			printf("%f ", layer_i->biases[j]);
		printf("\nlayer%d end.\n\n", i);
	}

	/* print the biases value layer by layer */
	for (int i = 0; i < net->layer_num - 1; i++) {
		struct link *link_i = net->links + i;
		printf("link%d : %d weights.\n", i, link_i->link_num);
		for (int j = 0; j < link_i->link_num; j++)
			if (j > 999) {
				printf("[and more]");
				break;
			} else {
				printf("%f ",link_i->weights[j]);
			}
		printf("\nlink%d end.\n\n", i);
	}
}

int net_save(struct net *net, char *path)
{
	FILE *file = fopen(path, "w");
	if (file == NULL)
		EXIT(EBADF);

	printf("Save the net to file...   ");

	fprintf(file, "layer num: %d \n", net->layer_num);
	fprintf(file, "activation mode: %d\n\n", net->activ_mode);

	/* save the biases value to file layer by layer */
	for (int i = 0; i < net->layer_num; i++) {
		struct layer *layer_i = net->layers + i;
		fprintf(file, "layer%d : %d biases.\n",
						i, layer_i->neurous_num);
		if (i == 0) {
			fprintf(file, "biases in first layer is useless.");
			fprintf(file, "\nlayer%d end.\n\n", i);
			continue;
		}
		for(int j = 0; j < layer_i->neurous_num; j++)
			fprintf(file, "%lf ", layer_i->biases[j]);
		fprintf(file, "\nlayer%d end.\n\n", i);
	}

	/* save the weights value to file layer by layer */
	for (int i = 0; i < net->layer_num - 1; i++) {
		struct link *link_i = net->links + i;
		fprintf(file, "link%d : %d weights.\n", i, link_i->link_num);
		for (int j = 0; j < link_i->link_num; j++)
			fprintf(file, "%lf ",link_i->weights[j]);
		fprintf(file, "\nlink%d end.\n\n", i);
	}

	fclose(file);
	printf("Done.\n");
}

int net_load(struct net *net, char *path)
{
	FILE *file = fopen(path, "r");
	if (file == NULL)
		EXIT(ENOENT);

	printf("Load the net from file...   ");
	
	/* check the layer num */
	int layer_num, ret;
	ret = fscanf(file, "layer num: %d \n", &layer_num);
	if(ret != 1)
		goto wrong_format;
	if(layer_num != net->layer_num)
		goto incompatiable;
	
	/* set the activ_mode of net */
	int activ_mode;
	ret = fscanf(file, "activation mode: %d\n\n",&activ_mode);
	if(ret != 1)
		goto wrong_format;
	net->activ_mode = activ_mode;
	switch (net->activ_mode) {
		case sigmoid:
			net->activ_func = sigmoid_func;
			net->fb_func = sigmoid_prime_func;
			break;
	}
	
	/* read biases from the file layer by layer*/		
	for (int i = 0; i < layer_num; i++) {
		ret = 1;
		int neu_num;
		struct layer *layer_i = net->layers + i;
		ret *= fscanf(file, "layer%d : %d biases.\n", &i, &neu_num); 
		if(neu_num != layer_i->neurous_num)
			goto incompatiable;
		if (i == 0) {
			ret += fscanf(file,
					"biases in first layer is useless.");
			ret *= fscanf(file, "\nlayer%d end.\n\n", &i);
			continue;
		}
		for(int j = 0; j < layer_i->neurous_num; j++)
			ret *= fscanf(file, "%lf ", layer_i->biases + j);
		ret *= fscanf(file, "\nlayer%d end.\n\n", &i);

		if(ret != 2)
			goto wrong_format;
	}

	
	/* read weights from the file layer by layer*/		
	for (int i = 0; i < net->layer_num - 1; i++) {
		ret = 1;
		int link_num;
		struct link *link_i = net->links + i;
		ret *= fscanf(file, "link%d : %d weights.\n", &i, &link_num);
		if(link_num != link_i->link_num)
			goto incompatiable;
		for (int j = 0; j < link_i->link_num; j++)
			ret *= fscanf(file, "%lf ", link_i->weights + j);
		ret *= fscanf(file, "\nlink%d end.\n\n", &i);
		
		if(ret != 2)
			goto wrong_format;
	}

	fclose(file);
	printf("Done.\n");
	return 0;

wrong_format:
	printf("Failed. Is the file created by 'net_save' function?\n");
	EXIT(ESPIPE);
incompatiable:
	printf("Failed. The file is incompatible with the net.\n");
	EXIT(-1);
}

int net_activ_save(struct net *net, char *path)
{
	FILE *file = fopen(path, "w");
        if (file == NULL)
                EXIT(EBADF);
	printf("Save the neurous to file...   ");

	for (int i = 0; i < net->layer_num; i++) {
                struct layer *layer_i = net->layers + i;
                fprintf(file, "layer%d : %d neurous.\n",
						i, layer_i->neurous_num);
                for(int j = 0; j < layer_i->neurous_num; j++)
                        fprintf(file, "%f ", layer_i->activ[j]);
                fprintf(file, "\nlayer%d end.\n\n", i);
        }

	fclose(file);
        printf("Done.\n");
}


double sigmoid_func(double x)
{
        return 1.0 / (1.0 + exp(-x));
}

/* Derivative of the sigmoid function. */
double sigmoid_prime_func(double x)
{
	double temp = (1.0+exp(-x));
        return exp(-x) / (temp * temp);
}


void net_init(struct net *net, enum activ_mode activ_mode)
{
	printf("Start initialize the neural network...  ");
	clock_t start_time = clock();

	/* Initialize the biases and weights */

	for (int i = 1; i < net->layer_num; i++) {
		struct layer *layer_i = net->layers + i;
		for (int j = 0; j < layer_i->neurous_num; j++)
			layer_i->biases[j] = guass_rand();
	}

	for (int i = 0; i < net->layer_num - 1; i++) {
		struct link *link_i = net->links + i;
		for (int j = 0; j < link_i->link_num; j++)
			link_i->weights[j] = guass_rand();
	}

	/* initialize the activi and fb_table */
	net->activ_mode = activ_mode;
	
	switch (net->activ_mode) {
		case sigmoid:
			net->activ_func = sigmoid_func;
			net->fb_func = sigmoid_prime_func;
			break;
	}
	
	clock_t end_time = clock();
	float used_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
	printf("Done! Used time: %.2f S.\n", used_time);
}


void layer_forward(struct net *net, struct link *link)
{
	unsigned int input_num = link->prev->neurous_num;
	unsigned int output_num = link->next->neurous_num;

	double *prev_a = link->prev->activ;
	double *b = link->next->biases;
	double *w = link->weights;
	double *nt = link->next->nt;
	double *a = link->next->activ;

	memset(nt, 0, sizeof(double) * output_num);
	for (int i = 0; i < output_num; i++) {
		for (int j = 0; j < input_num; j++) {
			nt[i] += prev_a[j] * w[i * input_num + j];	
		}
		a[i] = net->activ_func(nt[i] + b[i]);
	}
}

void propagate(struct net *net)
{	
	for (int i = 0; i < net->layer_num -1; i++) 
		layer_forward(net, net->links + i);
}

void net_cacu_err(struct net *net, double **b_err, double **w_err)
{
	/* layer.activ is used to save errors in this function */
	int layer_num = net->layer_num;
	
	for (int i = layer_num - 1; i > 0; i--) {
		int neu_num = net->layers[i].neurous_num;
		int prev_neu_mum = net->layers[i-1].neurous_num;
		struct layer *l = net->layers + i;
		double *link_w = (net->links + i - 1)->weights;
		struct layer *prev_l = net->layers + i - 1;
		for (int j = 0; j < neu_num; j++) {
			/* caculate the error of biases and save it to b_err*/
			l->activ[j] *= net->activ_func(l->nt[j]);
			b_err[i-1][j] += l->activ[j];

			/* caculate the error of weights and save it to w_err*/
			for (int k = 0; k < prev_neu_mum; k++) {
				int w_offset = prev_neu_mum * j + k;
				w_err[i-1][w_offset] +=
						l->activ[j] * prev_l->activ[k];
			}
		}

		if(i > 1) {
			/* trans the err to front layer */
			memset(prev_l->activ, 0, sizeof(double) * prev_neu_mum);
			for (int j = 0; j <  prev_neu_mum; j++) {
				for (int k = 0; k < neu_num; k++) {
					int w_offset = prev_neu_mum * k + j;
					prev_l->activ[j] += 
						link_w[w_offset] * l->activ[k];
				}
				prev_l->activ[j] *= 
						net->fb_func(prev_l->activ[j]);
			}
		}	
	}
}

void net_update(struct net *net, double **b_err, double **w_err, double speed)
{	
	int layer_num = net->layer_num;
	for(int i = 1; i <layer_num; i++) {
		/* updata biases */
		int b_len = net->layers[i].neurous_num;
		double *biases_i = net->layers[i].biases;
		double *b_err_i = b_err[i - 1];
		array_wsub(biases_i, b_err_i, biases_i, speed, b_len);

		/* updata weights */
		int w_len = net->layers[i - 1].neurous_num * 
						net->layers[i].neurous_num;
		double *weights_i = net->links[i - 1].weights;
		double *w_err_i = w_err[i -1];
		array_wsub(weights_i, w_err_i, weights_i, speed, w_len);
	}
}

int batch_train(struct net *net, struct data_pack *data, const double *target,
		unsigned int *index, unsigned int batch_size, double speed)
{
	unsigned int layer_num = net->layer_num;
	unsigned int input_num = net->layers[0].neurous_num;
	double *input = net->layers[0].activ;

	/* biases in first layer is useless. */
	double **b_err = malloc(sizeof(double *) * (layer_num - 1));  	
	double **w_err = malloc(sizeof(double *) * (layer_num - 1));
	if (input == NULL || w_err == NULL || b_err == NULL)
		EXIT(ENOMEM);

	for (int i = 0; i < layer_num - 1; i++) {
		unsigned int neu_num = net->layers[i + 1].neurous_num;
		unsigned int link_num = net->links[i].link_num;
		b_err[i] = calloc(neu_num, sizeof(double));
		w_err[i] = calloc(link_num, sizeof(double));
		if (b_err[i] == NULL || w_err[i] == NULL)
			EXIT(ENOMEM);
	}
	
	/*
	 * propagate forward and caculate the output error, then caculate 
	 * and save the errors of biases and weights to b_err and w_err.
	 */
	double *output = net->layers[layer_num - 1].activ;
	unsigned int out_num = net->layers[layer_num - 1].neurous_num;
	for (int i = 0; i < batch_size; i++) {
		int label = net_load_mnist(data, index[i], input, input_num);
		propagate(net);
		double *goal = (double *)target + out_num * label;
		array_sub(output, goal, output, out_num);
		net_cacu_err(net, b_err, w_err);
	}

	net_update(net, b_err, w_err, speed);
	
	for (int i = 0; i < layer_num - 1; i++) {
		free(b_err[i]);
		free(w_err[i]);
	}
	free(b_err);
	free(w_err);	
}


void rand_index(unsigned int *index_table, unsigned int len)
{
	unsigned int temp;
	for(int i = 0; i < len; i++)
		index_table[i] = i;

	for(int i = 0; i < len; i++) {
		srand(clock());
		int index1 = rand() % len;
		temp = index_table[index1];

		srand(clock());
		int index2 = rand() % len;
		index_table[index1] = index_table[index2];
		index_table[index2] = temp;
	}
}

void net_train(struct net *net, struct data_pack *data, const double *target, 
		unsigned int batch_size, double speed, unsigned int times)
{
	unsigned int data_num = data->img_num;
	if(batch_size > data_num)
		EXIT(EINVAL);

	
	unsigned int *index_table = 
			(unsigned int *)malloc(data_num * sizeof(unsigned int));
	if(index_table == NULL)
		EXIT(ENOMEM);

	printf("Start training the neural networks...\n");
	clock_t start_time = clock();
	
	printf("loop:batch of (%d):(%d)\n", times, data_num / batch_size);
	for (int t = 0; t < times; t ++) {
		rand_index(index_table, data_num);
		
		for (int i = 0; i <= data_num - batch_size; i += batch_size) {
			batch_train(net, data, target, index_table + i,
					batch_size, speed);
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
			printf(" %d : %d", t + 1, i / batch_size + 1);
		}
	}
	
	clock_t end_time = clock();
	double used_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
	printf("\nTraining completed! Used time: %.2f S.\n", used_time);
	free(index_table);
}

int net_work(struct net *net, struct data_pack *data, int index, bool detail)
{
	double *input = net->layers[0].activ;
	unsigned int input_num = net->layers[0].neurous_num;
	int label = net_load_mnist(data, index, input, input_num);
	propagate(net);
	
	double *output = net->layers[net->layer_num - 1].activ;
	unsigned int out_num = net->layers[net->layer_num - 1].neurous_num;
	double max = 0.0;
	int max_label = 0;
	for (int i = 0; i < out_num; i++) {
		if (output[i] > max) {
			max = output[i];
			max_label = i;
		}
		if (detail == true)
			printf("out%d : %f\n", i, output[i]);
	}
	return max_label;
}

float net_pack_test(struct net *net, struct data_pack *data)
{
	unsigned int data_num = data->img_num;
	unsigned int right = 0;
	int label, out;

	printf("Start testing all data in the data package...\n");
	printf("right/tested:\n");
	clock_t start_time = clock();
	for (int i = 0; i < data_num; i++) {
		label = data->imgs[i].label;
		out = net_work(net, data, i, false);
		if (out == label)
			right ++;
		printf("\b\b\b\b\b\b\b\b\b\b\b\b\b");
		printf("%d/%d", right, i + 1);
	}
	
	clock_t end_time = clock();
	float used_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;
	printf("\nTesting completed! Used time: %.2f S.\n", used_time);
	
	return (float)right / data_num;
}
