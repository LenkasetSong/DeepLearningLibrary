namespace DeepLearningLibrary.Networks
{
    using System;
    using System.Linq;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class DBNNetwork : Network
    {
        public new RBMLayer[] Layers
        {
            get
            {
                return Array.ConvertAll(layers, l => (RBMLayer)l);
            }
        }

        public DBNNetwork(IActivationFunction function, int inputsCount, params int[] neuronsCount)
			:base(inputsCount, neuronsCount.Length)
        {
            for (int i = 0; i < LayersCount; i++)
            {
                layers[i] = new RBMLayer(
                        neuronsCount[i],
                        (i == 0) ? inputsCount : neuronsCount[i - 1],
                        function);
            }
        }
    }
}
