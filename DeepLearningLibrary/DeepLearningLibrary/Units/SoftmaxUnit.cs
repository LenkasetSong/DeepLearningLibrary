namespace DeepLearningLibrary.Units
{
    using System;
    using DeepLearningLibrary.Utility.Common;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class SoftmaxUnit : Unit
    {
        protected int inputsCount = 0;

        protected double[] thetas = null;

        protected double output = 0;

        protected static Random rand = new Random((int)DateTime.Now.Ticks);

        protected static DoubleRange randRange = new DoubleRange(0.0, 1.0);

        public static Random RandGenerator
        {
            get { return rand; }
            set
            {
                if (value != null)
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

        public double[] Thetas
        {
            get { return thetas; }
            set { this.thetas = value; }
        }

        public SoftmaxUnit(int inputsCount)
        {
            this.inputsCount = inputsCount;
            thetas = new double[inputsCount];

            this.Randomize();
        }

        public override void Randomize()
        {
            double d = randRange.Length;

            for (int i = 0; i < inputsCount; i++)
            {
                thetas[i] = rand.NextDouble() * d + randRange.Min;
            }
        }

        public override double Compute(double[] input)
        {
            for(int i = 0; i < input.Length; i++)
            {
                output += thetas[i] * input[i];
            }

            return (output = Math.Exp(output));
        }
    }
}
