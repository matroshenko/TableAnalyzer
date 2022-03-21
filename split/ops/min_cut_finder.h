#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

using std::vector;
using std::unordered_map;
using std::pair;

struct PairHash {
public:
  template <typename T, typename U>
  std::size_t operator()(const pair<T, U> &x) const
  {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

class MinCutFinder {
public:
    typedef unordered_map<pair<int, int>, int, PairHash> TCapacity;

    MinCutFinder(const vector<vector<int>>& _graph, 
        const TCapacity& _capacity);

    // Finds min cut between source vertex s and destination vertex t.
    vector<bool> Find(int s, int t) const;

private:
    vector<vector<int>> graph;
    TCapacity capacity;

    bool bfs(int s, int t, const TCapacity& flow, vector<int>& parent) const;
    void dfs(int s, const TCapacity& flow, vector<bool>& visited) const;
};