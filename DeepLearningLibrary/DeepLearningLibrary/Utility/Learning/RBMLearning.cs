namespace DeepLearningLibrary.Utility.Learning
{
    using System;
    using DeepLearningLibrary.Networks;
    using DeepLearningLibrary.Neurons;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class RBMLearning : IUnsupervisedLearning
    {
        private RBMLayer rbm;

        private double learningRate = 0.1;

        private double momentum = 0.0;

        private double[] Q = null;

        private double[] P = null;

        private double[] weightsUpdates = null;

        private double[] h1 = null;

        private double[] h2 = null;



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

        public RBMLearning(RBMLayer rbm)
        {
            this.rbm = rbm;

            Q = new double[rbm.InputsCount];
            P = new double[rbm.NeuronsCount];
        }

        public double Run(double[] input)
        {
            
        }

        private void CalculateUpdate(double[] input)
        {
            Q = rbm.ComputeDown(input);


        }
    }
}
