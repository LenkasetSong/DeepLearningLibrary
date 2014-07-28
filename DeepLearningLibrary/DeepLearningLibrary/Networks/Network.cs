namespace DeepLearningLibrary.Networks
{
    using System;
    using DeepLearningLibrary.Layers;

    public abstract class Network
    {
        protected int inputsCount;

        protected int layersCount;

        protected Layer[] layers = null;

        protected double[] output = null;

        public int InputsCount
        {
            get { return inputsCount; }
        }

        public int LayersCount
        {
            get { return layersCount; }
        }

        public Layer[] Layers
        {
            get { return layers; }
        }

        protected Network(int inputsCount, int layersCount)
        {
            this.inputsCount = Math.Max(1, inputsCount);
            this.layersCount = Math.Max(1, layersCount);
            layers = new Layer[this.layersCount];
        }

        public virtual double[] Compute(double[] input)
        {
            output = input;

            foreach(Layer layer in layers)
            {
                output = layer.Compute(output);
            }

            return output;
        }

        public virtual void Randomize()
        {
            foreach(Layer layer in layers)
            {
                layer.Randomize();
            }
        }
    }
}
