namespace DeepLearningLibrary.Utility.Learning
{
    using System;

    public interface IUnsupervisedLearning
    {
        double Run(double[] input);

        double RunEpoch(double[][] input);
    }
}
