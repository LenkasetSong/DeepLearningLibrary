namespace DeepLearningLibrary.Utility.Learning
{
    using System;
    using DeepLearningLibrary.Networks;
    using DeepLearningLibrary.Units;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class RBMLearning : IUnsupervisedLearning
    {
        private RBMLayer rbm;

        private double learningRate = 0.1;

        private double momentum = 0.0;

        private int k = 1;

        private double[] Q = null;

        private double[] P = null;

        private double[] weightsUpdates = null;

        private double[] h1 = null;

        private double[] h2 = null;

        private double[] x1 = null;

        private double[] x2 = null;

        private static Random random;

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

        public int K
        {
            get { return k; }
            set { k = value; }
        }

        public RBMLearning(RBMLayer rbm)
        {
            this.rbm = rbm;

            Q = new double[rbm.InputsCount];
            P = new double[rbm.NeuronsCount];

            random = new Random((int)DateTime.Now.Ticks);
        }

        public double Run(double[] input)
        {
            this.CalculateUpdate(input);

            this.UpdateRBM();

            return 0;
        }

        public double RunEpoch(double[][] input)
        {
            double error = 0.0;

            // run learning procedure for all samples
            for (int i = 0, n = input.Length; i < n; i++)
            {
                Console.WriteLine(i);
                error += this.Run(input[i]);
            }

            // return summary error
            return error;
        }

        private double[] CalculateUpdate(double[] input)
        {
            x1 = input;

            Q = rbm.ComputeDown(input);

            h1 = this.Sampling(Q);

            h2 = h1;

            for (int i = 0; i < k; i++)
            {
                x2 = rbm.ComputeUp(h2);

                Q = rbm.ComputeDown(x2);

                h2 = this.Sampling(Q);
            }

            return x2;
        }

        private void UpdateRBM()
        {
            for (int i = 0; i < rbm.NeuronsCount; i++)
            {
                ActivationNeuron neuron = rbm.Neurons[i];

                for (int j = 0; j < neuron.InputsCount; j++)
                {
                    neuron.Weights[j] = neuron.Weights[j] + learningRate * (x1[j] * h1[i] - Q[i] * x2[j]);

                    rbm.Neurons_[j].Weights[i] = neuron.Weights[j];
                }
            }

            for (int i = 0; i < rbm.NeuronsCount; i++)
            {
                rbm.Neurons[i].Threshold = rbm.Neurons[i].Threshold + learningRate * (h1[i] - Q[i]);
            }

            for (int i = 0; i < rbm.InputsCount; i++)
            {
                rbm.Neurons_[i].Threshold = rbm.Neurons_[i].Threshold + learningRate * (x1[i] - x2[i]);
            }

        }

        private double[] Sampling(double[] prob)
        {
            double[] result = new double[prob.Length];

            for (int i = 0; i < prob.Length; i++ )
            {
                if (random.NextDouble() > prob[i])
                {
                    result[i] = 1;
                }
                else
                {
                    result[i] = 0;
                }
            }

            return result;
        }
    }
}
