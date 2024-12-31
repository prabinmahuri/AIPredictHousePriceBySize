using System;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program {
    //install Microsoft ML.NET with following command in the termonal
    //dotnet add package Microsoft.ML

    // define data structure
    public class HouseData {
        [LoadColumn(0)] public float Size;
        [LoadColumn(1)] public float Price;
    }

    public class Prediction {
        [ColumnName("Score")] public float Price;
    }
    static void Main(string[] args) {
        Console.WriteLine("\nMy First AI Model: Predict the price of house by size.\n");

        // Initialize the MLContext
        var context = new MLContext();

        // load sample data
        var data = new[] {
            new HouseData() { Size = 1.1F, Price = 1.2F },
            new HouseData() { Size = 1.9F, Price = 2.3F },
            new HouseData() { Size = 2.8F, Price = 3.0F },
            new HouseData() { Size = 3.4F, Price = 3.7F },
            new HouseData() { Size = 4.5F, Price = 4.9F },
            new HouseData() { Size = 5.0F, Price = 5.3F },
            new HouseData() { Size = 5.5F, Price = 5.8F }
        };

        // create model training data and covert the sample data to IDataView
        var trainingData = context.Data.LoadFromEnumerable(data);

        // create a training pipeleine for the model
        var pipeline = context.Transforms.Concatenate("Features", new[] { "Size" })
            .Append(context.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 500));

        // train the model
        var model = pipeline.Fit(trainingData);

        // output the prediction
        var size = new HouseData() { Size = 4.0F };
        var predictionEngine = context.Model.CreatePredictionEngine<HouseData, Prediction>(model);
        var price = predictionEngine.Predict(size);

        // output the prediction
        Console.WriteLine($"\nPredicted Price for Size {size.Size}: {price.Price:C}\n");

        // save the model
        context.Model.Save(model, trainingData.Schema, "modelHousePrice.zip");
        Console.WriteLine("\nModel Predict House Price By Size saved successfully!\n");
    }
}