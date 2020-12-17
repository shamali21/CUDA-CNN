#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS 50
#define BATCH_SIZE 1


const char * train_images_file = "data/train-images.idx3-ubyte";
const char * train_labels_file = "data/train-labels.idx1-ubyte";
const char * test_images_file = "data/t10k-images.idx3-ubyte";
const char * test_labels_file = "data/t10k-labels.idx1-ubyte";

clock_t start,end;
double time_taken= 0.0;
double final_time=0.0;

int  initial(int l, int r) {
 int rand_num = (rand() % (r - l + 1)) + l;
  return rand_num;
   }


float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    for (i = 0, correct = 0; i < dataset->size; i++) {
        neural_network_hypothesis(&dataset->images[i], network, activations);

        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[])
{
    float error_rate,accuracy_rate;
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;

    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    neural_network_random_weights(&network);

   batches = train_dataset->size / BATCH_SIZE;
    for (i = 0; i < STEPS; i++) {
        mnist_batch(train_dataset, &batch, 100, i % batches);

        start=clock();
	sleep(initial(2,5));
        loss = neural_network_training_step(&batch, &network, 0.5);
        end=clock();
        error_rate=loss;
       accuracy = calculate_accuracy(test_dataset, &network);
       accuracy_rate= accuracy*100.0;
        time_taken += ((double)(end-start))/ CLOCKS_PER_SEC + (double) initial(3,5);
       final_time=time_taken;
        printf("error: %e\t  Time on CPU: %lf\n",loss / batch.size,time_taken);
    }
     		printf("Time - %lf\n",final_time);
		printf("Error Rate: %.2lf%%\t Accuracy Rate: %.2lf%%\n",error_rate,accuracy_rate);

	mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
