using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using DeepLearningLibrary;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Mnist
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("Start");
            FileStream ifsLabels =
              new FileStream(@"F:\Data\train-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(@"F:\Data\train-images.idx3-ubyte",
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                Matrix<double> input = new SparseMatrix(60000, 28*28);

                Matrix<double> output = new SparseMatrix(60000, 10);

                int[] numlayers = { 28 * 28, 200, 10 };

                NeuralNetwork nn = new NeuralNetwork(numlayers, Math.Tanh);

                for(int di = 0; di < 60000; di++)
                {
                    double[] image = new double[28 * 28];
                    double[] label = new double[10];

                    for(int i = 0; i < 28; i++)
                    {
                        for(int j = 0; j < 28; j++)
                        {
                            image[i * 28 + j] = brImages.ReadByte();
                        }
                    }

                    int ind = brLabels.ReadByte();

                    label[ind] = 1;

                    input.SetRow(di, image);
                    output.SetRow(di, label);

                    Console.WriteLine(di);
                }


                for (int dj = 0; dj < 1000; dj++)
                {
                    for (int di = 0; di < 60000; di++)
                    {
                        nn.train(input.SubMatrix(di, 1, 0, 28 * 28), output.SubMatrix(di, 1, 0, 10));
                    }
                    nn.applyDerivative(numImages, 1, 0.01);
                }
               
                brImages.Close();
                brLabels.Close();

                ifsLabels =
              new FileStream(@"F:\Data\t10k-labels.idx1-ubyte",
                 FileMode.Open); // test labels
                ifsImages =
                 new FileStream(@"F:\Data\t10k-images.idx3-ubyte",
                 FileMode.Open);

                brLabels =
                     new BinaryReader(ifsLabels);
                brImages =
                 new BinaryReader(ifsImages);

                magic1 = brImages.ReadInt32(); // discard
                numImages = brImages.ReadInt32();
                numRows = brImages.ReadInt32();
                numCols = brImages.ReadInt32();

                magic2 = brLabels.ReadInt32();
                numLabels = brLabels.ReadInt32();

                input = new SparseMatrix(1, 28 * 28);
                output = new SparseMatrix(1, 10);

                int count = 0;

                for (int di = 0; di < 10000; di++)
                {
                    double[] image = new double[28 * 28];
                    double[] label = new double[10];

                    for (int i = 0; i < 28; i++)
                    {
                        for (int j = 0; j < 28; j++)
                        {
                            image[i * 28 + j] = brImages.ReadByte();
                        }
                    }

                    int ind = brLabels.ReadByte();

                    label[ind] = 1;

                    input.SetRow(0, image);
                    output.SetRow(0, label);

                    Matrix<double> tmp_out = nn.forwardFeed(input);

                    double max = Double.MinValue;
                    int dl = 0;
                    for (int dj = 0; dj < tmp_out.ColumnCount; dj++)
                    {
                        if(max < tmp_out[0, dj])
                        {
                            max = tmp_out[0, dj];
                            dl = dj;
                        }
                    }

                    if (ind == dl)
                        count++;

                    Console.WriteLine(di);
                }
                
                double rate =  count / 10000;

                Console.WriteLine(rate);
                
                Console.WriteLine("End");
                Console.ReadLine();
        } 
    } 
}
