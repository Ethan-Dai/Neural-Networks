#ifndef NN_NET_H
#define NN_NET_H

#include <stdbool.h>
#include "mnist.h"

struct layer {
	unsigned int neurous_num;
	double *nt;       //neurotransmitters
	double *activ;    //activation of neurous
	double *biases;
};

struct link {
	struct layer *prev;
	struct layer *next;
	unsigned int link_num;
	double *weights;
};

double sigmoid_func(double x);
double sigmoid_prime_func(double x);

enum activ_mode {
	sigmoid,
};

/*
 * struct net - neural networks descriptor
 * @layer_num:		the numbel of layers that the net contains
 * @layers:		layers in the net
 * @links:		the connections between TWO layers
 * @activ_mode:	the mode that neurous work on
 * @activ_func:	activation function
 * @fb_func:		derivative of the activation function, used to feedback.
 */
struct net {
	unsigned int layer_num;
	struct layer *layers;
	struct link *links;
	enum activ_mode activ_mode;
	double (*activ_func)(double);
	double (*fb_func)(double);
};


/* Allocate memory for the network */
int net_setup(struct net *net, const unsigned int layer_num,
		const unsigned int *neurous_num);

/* Free up memory used by the network */
void net_del(struct net *net);

/* Initialize the network with random values and set the activation function */
void net_init(struct net *net, enum activ_mode activ_mode);

/* Print the key data of the network */
void net_print(struct net *net);

/* Save the key data of the network */
int net_save(struct net *net, char *path);

/* Load the key data of the network from the file */
int net_load(struct net *net, char *path);

/* Train a neural network */
void net_train(struct net *net, struct data_pack *data, const double *target, 
		unsigned int batch_size, double speed, unsigned int times);

/* Test the network with a single data */
int net_work(struct net *net, struct data_pack *data, int index, bool detail);

/* Test the network with multiple sets of data */
float net_pack_test(struct net *net, struct data_pack *data);

/* Save network activation(error) data to a file, usually for debugging */
int net_activ_save(struct net *net, char *path);

#endif /* NN_NET_H */
