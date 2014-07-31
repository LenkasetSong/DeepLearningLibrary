namespace DeepLearningLibrary.Units
{
    using System;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class ActivationNeuron : Neuron
    {
        protected double threshold = 0.0f;

        protected IActivationFunction function = null;

        public double Threshold
        {
            get { return threshold; }
            set { threshold = value; }
        }

        public IActivationFunction ActivationFunction
        {
            get { return function; }
        }

        public ActivationNeuron(int inputs, IActivationFunction function)
            : base(inputs)
        {
            this.function = function;
        }

        public override void Randomize()
        {
            base.Randomize();

            threshold = rand.NextDouble() * (randRange.Length) + randRange.Min;
        }

        public override double Compute(double[] input)
        {
            // check for corrent input vector
            if (input.Length != InputsCount)
                throw new ArgumentException();

            // initial sum value
            z = 0.0;

            // compute weighted sum of inputs
            for (int i = 0; i < inputsCount; i++)
            {
                z += weights[i] * input[i];
            }
            z += threshold;

            return (output = function.Function(z));
        }
    }
}
