using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using DeepLearningLibrary.Networks;
using DeepLearningLibrary.Layers;
using DeepLearningLibrary.Utility.ActivationFunction;
using DeepLearningLibrary.Utility.Learning;

namespace Mnist
{
    class Program
    {
        static void Main()
        {
            int numImages = 100;
            
            double[][] result = new double[numImages][];

            double[][] labels = new double[numImages][];

            FileStream ifsPixels = new FileStream("F:/data/train-images.idx3-ubyte", FileMode.Open);
            FileStream ifsLabels = new FileStream("F:/data/train-labels.idx1-ubyte", FileMode.Open);

            BinaryReader brImages = new BinaryReader(ifsPixels);
            BinaryReader brLabels = new BinaryReader(ifsLabels);

            int magic1 = brImages.ReadInt32(); // stored as Big Endian
            magic1 = ReverseBytes(magic1); // convert to Intel format

            int imageCount = brImages.ReadInt32();
            imageCount = ReverseBytes(imageCount);

            int numRows = brImages.ReadInt32();
            numRows = ReverseBytes(numRows);
            int numCols = brImages.ReadInt32();
            numCols = ReverseBytes(numCols);

            int magic2 = brLabels.ReadInt32();
            magic2 = ReverseBytes(magic2);

            int numLabels = brLabels.ReadInt32();
            numLabels = ReverseBytes(numLabels);

            // each image
            for (int di = 0; di < numImages; ++di)
            {
                Console.WriteLine(di);

                result[di] = new double[784];
                labels[di] = new double[10];

                for (int i = 0; i < 784; ++i) // get 28x28 pixel values
                {
                    byte b = brImages.ReadByte();
                    result[di][i] = (double)b/255.0;
                }

                byte lbl = brLabels.ReadByte(); // get the label

                labels[di][lbl] = 1;
                
            } // each image

            SigmoidFunction function = new SigmoidFunction(0.01);

            DBNNetwork dbn = new DBNNetwork(function, 784, 100, 100);

            DBNLearning dbn_learn = new DBNLearning(dbn);

            dbn_learn.Epoch = 1;

            dbn_learn.RunEpoch(result);

            ActivationNetwork nn = new ActivationNetwork(function, 100, 10);

            BackPropagationLearning bp = new BackPropagationLearning(nn);

            for(int i = 0; i < numImages; i++)
            {
                Console.WriteLine(i);
                result[i] = dbn.Compute(result[i]);
            }

            double error = 1;

            bp.LearningRate = 1;

            for(int i = 0; i < 100 && error > 0.1; i++)
            {
                Console.WriteLine(error);
                error = bp.RunEpoch(result, labels);
            }

            int count = 0;

            for (int i = 0; i < numImages; i++)
            {
                double[] test = nn.Compute(result[i]);

                double max = 0;
                int ind = 0;
                int ans = 0;
                for(int j = 0; j < 10; j++)
                {
                    if(max < test[j])
                    {
                        ind = j;
                        max = test[j];
                    }

                    if(labels[i][j] == 1)
                    {
                        ans = j;
                    }
                }

                if(ind == ans)
                    count++;
            }

            Console.WriteLine(count);

            Console.ReadLine();
        }

        public static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }
    } 
}
