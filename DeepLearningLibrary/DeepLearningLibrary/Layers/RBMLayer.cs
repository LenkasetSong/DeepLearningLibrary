namespace DeepLearningLibrary.Layers
{
    using System;
    using System.Linq;
    using DeepLearningLibrary.Units;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class RBMLayer : ActivationLayer
    {
        protected Neuron[] neurons_ = null;

        protected double[] output_ = null;

        public ActivationNeuron[] Neurons_
        {
            get
            {
                return Array.ConvertAll(neurons_, n => (ActivationNeuron)n);
            }
        }

        public double[] Output_
        {
            get { return output_; }
        }

        public virtual double[] ComputeDown(double[] input)
        {
            for (int i = 0; i < neuronsCount; i++)
                output[i] = neurons[i].Compute(input);

            return output;
        }

        public virtual double[] ComputeUp(double[] input)
        {
            for (int i = 0; i < inputsCount; i++)
                output_[i] = neurons_[i].Compute(input);

            return output_;
        }

        public RBMLayer(int neuronsCount, int inputsCount, IActivationFunction function)
            :base(neuronsCount, inputsCount, function)
        {
            neurons_ = new Neuron[inputsCount];

            output_ = new double[inputsCount];

            for(int i = 0; i < inputsCount; i++)
            {
                neurons_[i] = new ActivationNeuron(neuronsCount, function);
            }
        }
    }
}
