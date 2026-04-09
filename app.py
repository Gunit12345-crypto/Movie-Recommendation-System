from model import recommend

print("🎬 Movie Recommendation System")

movie_name = input("Enter movie name: ")

try:
    results = recommend(movie_name)

    print("\nRecommended Movies:")
    for movie in results:
        print("👉", movie)

except:
    print("❌ Movie not found, try another one")
