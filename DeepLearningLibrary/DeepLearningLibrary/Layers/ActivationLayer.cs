namespace DeepLearningLibrary.Layers
{
    using System;
    using System.Linq;
    using DeepLearningLibrary.Units;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class ActivationLayer : Layer
    {
        public new ActivationNeuron[] Neurons
        {
            get
            {
                return Array.ConvertAll(neurons, n => (ActivationNeuron) n);
            }
        }

        public ActivationLayer(int neuronsCount, int inputsCount, IActivationFunction function)
            :base(neuronsCount, inputsCount)
        {
            for (int i = 0; i < neuronsCount; i++)
            {
                this.neurons[i] = new ActivationNeuron(inputsCount, function);
            }
        }
    }
}
