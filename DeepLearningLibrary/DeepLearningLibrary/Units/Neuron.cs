namespace DeepLearningLibrary.Units
{
    using System;
    using DeepLearningLibrary.Utility.Common;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public abstract class Neuron : Unit
    {
        protected int inputsCount = 0;

        protected double[] weights = null;

        protected double output = 0;

        protected double z = 0;

        protected static Random rand = new Random((int)DateTime.Now.Ticks);

        protected static DoubleRange randRange = new DoubleRange(0.0, 1.0);

        public static Random RandGenerator
        {
            get { return rand; }
            set
            {
                if(value != null)
                {
                    rand = value;
                }
            }
        }

        public static DoubleRange RandRange
        {
            get { return randRange; }
            set
            {
                if (value != null)
                {
                    randRange = value;
                }
            }
        }

        public int InputsCount
        {
            get { return inputsCount; }
        }

        public double Output
        {
            get { return output; }
        }

        public double Z
        {
            get { return z; }
        }

        public double[] Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        protected Neuron(int inputs)
        {
            this.inputsCount = inputs;
            Weights = new double[inputs];
            this.Randomize();
        }

        public virtual void Randomize()
        {
            double d = randRange.Length;
            
            for(int i = 0; i < inputsCount; i++)
            {
                weights[i] = rand.NextDouble() * d + randRange.Min;
            }
        }
    }
}
