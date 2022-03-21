#include <vector>
#include <unordered_map>
#include <utility>

using std::vector;
using std::unordered_map;
using std::pair;

class MinCutFinder {
public:
    MinCutFinder(const vector<vector<int>>& _graph, 
        const unordered_map<pair<int, int>, int>& _capacity);

    // Finds min cut between source vertex s and destination vertex t.
    vector<bool> Find(int s, int t) const;

private:
    vector<vector<int>> graph;
    unordered_map<pair<int, int>, int> capacity;

    bool bfs(int s, int t, const unordered_map<pair<int, int>, int>& flow, vector<int>& parent) const;
    void dfs(int s, const unordered_map<pair<int, int>, int>& flow, vector<bool>& visited) const;
};