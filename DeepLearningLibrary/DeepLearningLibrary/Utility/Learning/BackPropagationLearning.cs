namespace DeepLearningLibrary.Utility.Learning
{
    using System;
    using DeepLearningLibrary.Networks;
    using DeepLearningLibrary.Units;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class BackPropagationLearning : ISupervisedLearning
    {
        private ActivationNetwork network;

        private double learningRate = 0.1;

        private double momentum = 0.0;

        private double[][] neuronErrors = null;
       
        private double[][][] weightsUpdates = null;

        private double[][] thresholdsUpdates = null;

        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        public double Momentum
        {
            get { return momentum; }
            set
            {
                momentum = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        public BackPropagationLearning( ActivationNetwork network )
        {
            this.network = network;

            neuronErrors = new double[network.LayersCount][];
            weightsUpdates = new double[network.LayersCount][][];
            thresholdsUpdates = new double[network.LayersCount][];

            for (int i = 0, n = network.LayersCount; i < n; i++)
            {
                Layer layer = network.Layers[i];

                neuronErrors[i] = new double[layer.NeuronsCount];
                weightsUpdates[i] = new double[layer.NeuronsCount][];
                thresholdsUpdates[i] = new double[layer.NeuronsCount];

                for (int j = 0; j < layer.NeuronsCount; j++)
                {
                    weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }
        }

        public double Run(double[] input, double[] output)
        {
            // compute the network's output
            network.Compute(input);

            // calculate network error
            double error = this.CalculateError(output);

            // calculate weights updates
            this.CalculateUpdates(input);

            // update the network
            UpdateNetwork();

            return error;
        }

        public double RunEpoch(double[][] input, double[][] output)
        {
            double error = 0.0;

            // run learning procedure for all samples
            for (int i = 0, n = input.Length; i < n; i++)
            {
                error += Run(input[i], output[i]);
            }

            // return summary error
            return error;
        }

        private double CalculateError(double[] desiredOutput)
        {
            // current and the next layers
            ActivationLayer layer, layerNext;
            // current and the next errors arrays
            double[] errors, errorsNext;
            // error values
            double error = 0, e, sum;
            // neuron's output value
            double output;
            // layers count
            int layersCount = network.LayersCount;

            // assume, that all neurons of the network have the same activation function
            IActivationFunction function = network.Layers[0].Neurons[0].ActivationFunction;

            // calculate error values for the last layer first
            layer = network.Layers[layersCount - 1];
            errors = neuronErrors[layersCount - 1];

            for (int i = 0, n = layer.NeuronsCount; i < n; i++)
            {
                output = layer.Neurons[i].Output;
                // error of the neuron
                e = desiredOutput[i] - output;
                // error multiplied with activation function's derivative
                errors[i] = e * function.Derivative(layer.Neurons[i].Z);
                // squre the error and sum it
                error += (e * e);
            }

            // calculate error values for other layers
            for (int j = layersCount - 2; j >= 0; j--)
            {
                layer = network.Layers[j];
                layerNext = network.Layers[j + 1];
                errors = neuronErrors[j];
                errorsNext = neuronErrors[j + 1];

                // for all neurons of the layer
                for (int i = 0, n = layer.NeuronsCount; i < n; i++)
                {
                    sum = 0.0;
                    // for all neurons of the next layer
                    for (int k = 0, m = layerNext.NeuronsCount; k < m; k++)
                    {
                        sum += errorsNext[k] * layerNext.Neurons[k].Weights[i];
                    }
                    errors[i] = sum * function.Derivative(layer.Neurons[i].Z);
                }
            }

            // return squared error of the last layer divided by 2
            return error / 2.0;
        }

        private void CalculateUpdates(double[] input)
        {
            // current neuron
            ActivationNeuron neuron;
            // current and previous layers
            ActivationLayer layer, layerPrev;
            // layer's weights updates
            double[][] layerWeightsUpdates;
            // layer's thresholds updates
            double[] layerThresholdUpdates;
            // layer's error
            double[] errors;
            // neuron's weights updates
            double[] neuronWeightUpdates;
            // error value
            double error;

            layer = network.Layers[0];
            errors = neuronErrors[0];
            layerWeightsUpdates = weightsUpdates[0];
            layerThresholdUpdates = thresholdsUpdates[0];

            for (int i = 0, n = layer.NeuronsCount; i < n; i++)
            {
                neuron = layer.Neurons[i];
                error = errors[i];
                neuronWeightUpdates = layerWeightsUpdates[i];

                // for each weight of the neuron
                for (int j = 0, m = neuron.InputsCount; j < m; j++)
                {
                    // calculate weight update
                    neuronWeightUpdates[j] = learningRate * (
                        momentum * neuronWeightUpdates[j] +
                        (1.0 - momentum) * error * input[j]
                        );
                }

                // calculate treshold update
                layerThresholdUpdates[i] = learningRate * (
                    momentum * layerThresholdUpdates[i] +
                    (1.0 - momentum) * error
                    );
            }

            for (int k = 1, l = network.LayersCount; k < l; k++)
            {
                layerPrev = network.Layers[k - 1];
                layer = network.Layers[k];
                errors = neuronErrors[k];
                layerWeightsUpdates = weightsUpdates[k];
                layerThresholdUpdates = thresholdsUpdates[k];

                // for each neuron of the layer
                for (int i = 0, n = layer.NeuronsCount; i < n; i++)
                {
                    neuron = layer.Neurons[i];
                    error = errors[i];
                    neuronWeightUpdates = layerWeightsUpdates[i];

                    // for each synapse of the neuron
                    for (int j = 0, m = neuron.InputsCount; j < m; j++)
                    {
                        // calculate weight update
                        neuronWeightUpdates[j] = learningRate * (
                            momentum * neuronWeightUpdates[j] +
                            (1.0 - momentum) * error * layerPrev.Neurons[j].Output
                            );
                    }

                    // calculate treshold update
                    layerThresholdUpdates[i] = learningRate * (
                        momentum * layerThresholdUpdates[i] +
                        (1.0 - momentum) * error
                        );
                }
            }
        }

        private void UpdateNetwork()
        {
            // current neuron
            ActivationNeuron neuron;
            // current layer
            ActivationLayer layer;
            // layer's weights updates
            double[][] layerWeightsUpdates;
            // layer's thresholds updates
            double[] layerThresholdUpdates;
            // neuron's weights updates
            double[] neuronWeightUpdates;

            // for each layer of the network
            for (int i = 0, n = network.LayersCount; i < n; i++)
            {
                layer = network.Layers[i];
                layerWeightsUpdates = weightsUpdates[i];
                layerThresholdUpdates = thresholdsUpdates[i];

                // for each neuron of the layer
                for (int j = 0, m = layer.NeuronsCount; j < m; j++)
                {
                    neuron = layer.Neurons[j];
                    neuronWeightUpdates = layerWeightsUpdates[j];

                    // for each weight of the neuron
                    for (int k = 0, s = neuron.InputsCount; k < s; k++)
                    {
                        // update weight
                        neuron.Weights[k] += neuronWeightUpdates[k];
                    }
                    // update treshold
                    neuron.Threshold += layerThresholdUpdates[j];
                }
            }
        }
    }
}
