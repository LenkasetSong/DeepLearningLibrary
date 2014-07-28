namespace DeepLearningLibrary.Layers
{
    using System;
    using DeepLearningLibrary.Neurons;

    public abstract class Layer
    {
        protected int inputsCount = 0;

        protected int neuronsCount = 0;

        protected Neuron[] neurons = null;

        protected double[] output = null;

        public int InputsCount
        {
            get { return inputsCount; }
        }

        public int NeuronsCount
        {
            get { return neuronsCount; }
        }

        public double[] Ouput
        {
            get { return output; }
        }

        public Neuron[] Neurons
        {
            get { return neurons; }
        }

        protected Layer( int neuronsCount, int inputsCount )
		{
			this.inputsCount	= Math.Max( 1, inputsCount );
			this.neuronsCount	= Math.Max( 1, neuronsCount );
			output = new double[this.neuronsCount];
            neurons = new Neuron[this.neuronsCount];
		}

        public virtual double[] Compute(double[] input)
        {
            for (int i = 0; i < neuronsCount; i++)
                output[i] = neurons[i].Compute(input);

            return output;
        }

        public virtual void Randomize()
        {
            foreach (Neuron neuron in neurons)
                neuron.Randomize();
        }
    }
}
