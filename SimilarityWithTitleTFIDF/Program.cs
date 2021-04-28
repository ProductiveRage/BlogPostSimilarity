using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Catalyst;
using Catalyst.Models;
using HNSW.Net;
using Markdig;
using Mosaik.Core;
using Newtonsoft.Json;
using UID;

namespace SimilarityWithTitleTFIDF
{
    internal static class Program
    {
        private static async Task Main()
        {
            Console.WriteLine("Reading posts from GitHub repo..");
            var posts = await GetBlogPosts();

            Console.WriteLine("Parsing documents..");
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));
            var language = Language.English;
            var pipeline = Pipeline.For(language);
            var postsWithDocuments = posts
                .Select(post =>
                {
                    var document = new Document(NormaliseSomeCommonTerms(post.PlainTextContent), language)
                    {
                        UID = post.Title.Hash128()
                    };
                    pipeline.ProcessSingle(document);
                    return (Post: post, Document: document);
                })
                .ToArray(); // Call ToArray to force evaluation of the document processing now

            Console.WriteLine("Training FastText model..");
            var fastText = new FastText(language, version: 0, tag: "");
            fastText.Data.Type = FastText.ModelType.PVDM;
            fastText.Data.Loss = FastText.LossType.NegativeSampling;
            fastText.Data.IgnoreCase = true;
            fastText.Data.Epoch = 50;
            fastText.Data.Dimensions = 512;
            fastText.Data.MinimumCount = 1;
            fastText.Data.ContextWindow = 10;
            fastText.Data.NegativeSamplingCount = 20;
            fastText.Train(
                postsWithDocuments.Select(postsWithDocument => postsWithDocument.Document),
                trainingStatus: update => Console.WriteLine($" Progress: {update.Progress}, Epoch: {update.Epoch}")
            );

            Console.WriteLine("Training TF-IDF model..");
            var tfidf = new TFIDF(pipeline.Language, version: 0, tag: "");
            await tfidf.Train(postsWithDocuments.Select(postWithDocument => postWithDocument.Document));

            Console.WriteLine("Getting average TF-IDF weights per word..");
            var tokenValueTFIDF = new Dictionary<string, List<float>>(StringComparer.OrdinalIgnoreCase);
            foreach (var doc in postsWithDocuments.Select(postWithDocument => postWithDocument.Document))
            {
                tfidf.Process(doc);
                foreach (var sentence in doc)
                {
                    foreach (var token in sentence)
                    {
                        if (!tokenValueTFIDF.TryGetValue(token.Value, out var freqs))
                        {
                            freqs = new();
                            tokenValueTFIDF.Add(token.Value, freqs);
                        }
                        freqs.Add(token.Frequency);
                    }
                }
            }
            var averagedTokenValueTFIDF = tokenValueTFIDF.ToDictionary(
                entry => entry.Key,
                entry => entry.Value.Average(), StringComparer.OrdinalIgnoreCase
            );

            Console.WriteLine("Building recommendations..");

            // Combine the blog post data with the FastText-generated vectors
            var results = fastText
                .GetDocumentVectors()
                .Select(result =>
                {
                    // Each document vector instance will include a "token" string that may be mapped back to the
                    // UID of the document for each blog post. If there were a large number of posts to deal with
                    // then a dictionary to match UIDs to blog posts would be sensible for performance but I only
                    // have a 100+ and so a LINQ "First" scan over the list will suffice.
                    var uid = UID128.Parse(result.Token);
                    var postForResult = postsWithDocuments.First(
                        postWithDocument => postWithDocument.Document.UID == uid
                    );
                    return (UID: uid, result.Vector, postForResult.Post);
                })
                .ToArray(); // ToArray since we enumerate multiple times below

            // Construct a graph to search over, as described at
            // https://github.com/curiosity-ai/hnsw-sharp#how-to-build-a-graph
            var graph = new SmallWorld<(UID128 UID, float[] Vector, BlogPost Post), float>(
                distance: (to, from) => CosineDistance.NonOptimized(from.Vector, to.Vector),
                DefaultRandomGenerator.Instance,
                new() { M = 15, LevelLambda = 1 / Math.Log(15) }
            );
            graph.AddItems(results);

            const int maximumNumberOfResultsToReturn = 3;
            var postsWithSimilarResults = results
                .Select(result =>
                {
                    // Request that the KNNSearch operate over all documents because we can't take the top {n}
                    // until we've combined the ordering with the title TFIDF proximity values
                    var similarResults = graph
                        .KNNSearch(result, postsWithDocuments.Length)
                        .Where(similarResult => similarResult.Item.UID != result.UID);

                    var tokenValuesInTitle =
                        GetAllTokensForText(NormaliseSomeCommonTerms(result.Post.Title), pipeline)
                            .Select(token => token.Value)
                            .ToHashSet(StringComparer.OrdinalIgnoreCase);

                    return new
                    {
                        result.Post,
                        Similar = similarResults
                            .Select(similarResult => new
                            {
                                similarResult.Item.Post,
                                similarResult.Distance,
                                ProximityByTitleTFIDF = GetProximityByTitleTFIDF(
                                    NormaliseSomeCommonTerms(similarResult.Item.Post.Title),
                                    tokenValuesInTitle,
                                    averagedTokenValueTFIDF,
                                    pipeline
                                )
                            })
                            .OrderByDescending(similarResult => similarResult.ProximityByTitleTFIDF)
                            .ThenBy(similarResult => similarResult.Distance)
                            .Take(maximumNumberOfResultsToReturn)
                            .ToArray()
                    };
                })
                .OrderBy(result => result.Post.Title, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            foreach (var postWithSimilarResults in postsWithSimilarResults)
            {
                Console.WriteLine();
                Console.WriteLine(postWithSimilarResults.Post.Title);
                foreach (var similarResult in postWithSimilarResults.Similar.OrderBy(other => other.Distance))
                    Console.WriteLine($"{similarResult.ProximityByTitleTFIDF:0.000} {similarResult.Distance:0.000} {similarResult.Post.Title}");
            }

            Console.WriteLine();
            Console.WriteLine("Done! Press [Enter] to terminate..");
            Console.ReadLine();
        }

        private static float GetProximityByTitleTFIDF(string similarPostTitle, HashSet<string> tokenValuesInInitialPostTitle, Dictionary<string, float> averagedTokenValueTFIDF, Pipeline pipeline)
        {
            return GetAllTokensForText(similarPostTitle, pipeline)
                .Where(token => tokenValuesInInitialPostTitle.Contains(token.Value))
                .Sum(token =>
                {
                    var tfidfValue = averagedTokenValueTFIDF.TryGetValue(token.Value, out var score) ? score : 0;
                    if (tfidfValue <= 0)
                    {
                        // Ignore any tokens that report a negative impact (eg. punctuation or really common words like "in")
                        return 0;
                    }
                    return tfidfValue;
                });
        }

        private static IEnumerable<IToken> GetAllTokensForText(string text, Pipeline pipeline)
        {
            var doc = new Document(text, pipeline.Language);
            pipeline.ProcessSingle(doc);
            return doc.SelectMany(sentence => sentence);
        }
        
        private static string NormaliseSomeCommonTerms(string text) => text
            .Replace(".NET", "NET", StringComparison.OrdinalIgnoreCase)
            .Replace("Full Text Indexer", "FullTextIndexer", StringComparison.OrdinalIgnoreCase)
            .Replace("Bridge.net", "BridgeNET", StringComparison.OrdinalIgnoreCase)
            .Replace("React", "ReactJS");

        private static async Task<IEnumerable<BlogPost>> GetBlogPosts()
        {
            // Note: The GitHub API is rate limited quite severely for non-authenticated apps, so we just
            // call it once for the list of files and then retrieve them all further down via the Download
            // URLs (which don't count as API calls). Still, if you run this code repeatedly and start
            // getting 403 "rate limited" responses then you might have to hold off for a while.
            string namesAndUrlsJson;
            using (var client = new WebClient())
            {
                // The API refuses requests without a User Agent, so set one before calling (see
                // https://docs.github.com/en/rest/overview/resources-in-the-rest-api#user-agent-required)
                client.Headers.Add(HttpRequestHeader.UserAgent, "ProductiveRage Blog Post Example");
                namesAndUrlsJson = await client.DownloadStringTaskAsync(new Uri(
                    "https://api.github.com/repos/ProductiveRage/Blog/contents/Blog/App_Data/Posts?ref=master"
                ));
            }

            // Deserialise the response into an array of entries that have Name and Download_Url properties
            var namesAndUrls = JsonConvert.DeserializeAnonymousType(
                namesAndUrlsJson,
                new[] { new { Name = "", Download_Url = (Uri)null } }
            );

            return await Task.WhenAll(namesAndUrls
                .Select(entry =>
                {
                    var fileNameSegments = Path.GetFileNameWithoutExtension(entry.Name).Split(",");
                    if (fileNameSegments.Length < 8)
                        return default;
                    if (!int.TryParse(fileNameSegments[0], out var id))
                        return default;
                    var dateContent = string.Join(",", fileNameSegments.Skip(1).Take(6));
                    if (!DateTime.TryParseExact(dateContent, "yyyy,M,d,H,m,s", default, default, out var date))
                        return default;
                    return (PostID: id, PublishedAt: date, entry.Download_Url);
                })
                .Where(entry => entry != default)
                .Select(async entry =>
                {
                    // Read the file content as markdown and parse into plain text (the first line of which
                    // will be the title of the post)
                    string markdown;
                    using (var client = new WebClient())
                    {
                        markdown = await client.DownloadStringTaskAsync(entry.Download_Url);
                    }
                    var plainText = Markdown.ToPlainText(markdown);
                    var title = plainText.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n').First();
                    return new BlogPost(entry.PostID, title, plainText, entry.PublishedAt);
                })
            );
        }

        private sealed class BlogPost
        {
            public BlogPost(int id, string title, string plainTextContent, DateTime publishedAt)
            {
                ID = id;
                Title = !string.IsNullOrWhiteSpace(title)
                    ? title
                    : throw new ArgumentException("may not be null, blank or whitespace-only");
                PlainTextContent = !string.IsNullOrWhiteSpace(plainTextContent)
                    ? plainTextContent
                    : throw new ArgumentException("may not be null, blank or whitespace-only");
                PublishedAt = publishedAt;
            }

            public int ID { get; }
            public string Title { get; }
            public string PlainTextContent { get; }
            public DateTime PublishedAt { get; }
        }
    }
}
