namespace DeepLearningLibrary.Networks
{
    using System;
    using System.Linq;
    using DeepLearningLibrary.Layers;
    using DeepLearningLibrary.Utility.ActivationFunction;

    public class ActivationNetwork : Network
    {
        public new ActivationLayer[] Layers
        {
            get
            {
                return Array.ConvertAll(layers, l => (ActivationLayer) l);
            }
        }

        public ActivationNetwork(IActivationFunction function, int inputsCount, params int[] neuronsCount)
            :base(inputsCount, neuronsCount.Length)
        {
            for (int i = 0; i < layersCount; i++)
            {
                layers[i] = new ActivationLayer(
                    neuronsCount[i],
                    (i == 0) ? inputsCount : neuronsCount[i - 1],
                    function);
            }
        }
    }
}
