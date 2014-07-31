namespace DeepLearningLibrary.Utility.Learning
{
    using System;
    using DeepLearningLibrary.Networks;
    using DeepLearningLibrary.Units;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class DBNLearning : IUnsupervisedLearning
    {
        private DBNNetwork dbn;

        private RBMLearning[] rbm_learnings;

        private double learningRate = 0.1;

        private double momentum = 0.0;

        private int k = 1;

        private int epoch = 1;

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

        public int Epoch
        {
            get { return epoch; }
            set { epoch = value; }
        }

        public RBMLearning[] Rbm_learnings
        {
            get { return rbm_learnings; }
            set { rbm_learnings = value; }
        }

        public DBNLearning(DBNNetwork dbn)
        {
            this.dbn = dbn;

            rbm_learnings = new RBMLearning[dbn.LayersCount];
        }

        public double Run(double[] input)
        {
            throw new NotImplementedException();
        }

        public double RunEpoch(double[][] input)
        {
            double error = 0.0;

            double[][] data = new double[input.Length][];

            for(int i = 0; i < data.Length; i++)
            {
                data[i] = (double[]) input[i].Clone();
            }

            for (int i = 0; i < rbm_learnings.Length; i++)
            {
                Console.WriteLine(i);

                RBMLayer rbm = dbn.Layers[i];

                rbm_learnings[i] = new RBMLearning(rbm);

                for (int j = 0; j < epoch; j++)
                {
                    rbm_learnings[i].RunEpoch(data);
                }

                for (int j = 0; j < input.Length; j++)
                {
                    data[j] = rbm.ComputeDown(data[j]);
                }
            }

            return error;
        }
    }
}
