namespace DeepLearningLibrary.Utility.ActivationFunction
{
    using System;

    public interface IActivationFunction
    {
        double Function(double x);

        double Derivative(double x);

        double Derivative2(double y);
    }
}
