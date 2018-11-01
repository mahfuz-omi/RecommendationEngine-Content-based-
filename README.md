# RecommendationEngine(Content-based)

This is a simple Content based Movie Recommendation Engine.
I have adopted 2 ideas in building Recommendation System such as: 
1) Only Plot Keywords based(Recommendation_engine_test.py)  
2) Consider Other features like Plot Keywords, genres,director, actor,imdb score etc(Recommendation_engine_weighted.py).


I have used TF-IDF Vectorizer provided by sklearn library to create vector for plot keywords and genres. The selected movie vector is then compared with other vectors by Cosine Similarity Score and the similar movies are then recommended by the system.

There are several controls such as: Importance of actor,importance of director, importance of plot, importance of genres etc. Users can easily change these parameters and explore the essence of Movie Recommendation.

Happy Coding :p
