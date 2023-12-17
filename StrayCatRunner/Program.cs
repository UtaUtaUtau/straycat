using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

class Program
{
    static readonly HttpClient client = new HttpClient();

    static async Task Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("No arguments provided.");
            Environment.Exit(1);
        }

        try
        {
            var postFields = string.Join(" ", args);
            System.Console.WriteLine($"Sending: {postFields}");
            var content = new StringContent(postFields, Encoding.UTF8, "application/x-www-form-urlencoded");

            HttpResponseMessage response = await client.PostAsync("http://127.0.0.1:8572", content); // Replace with your URL

            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine("Success: Server returned 200 OK");
            }
            else
            {
                Console.Error.WriteLine($"Error: Server returned {response.StatusCode}");
            }
        }
        catch (HttpRequestException e)
        {
            Console.Error.WriteLine($"Request exception: {e.Message}\nIs straycat_server running?");
        }
    }
}