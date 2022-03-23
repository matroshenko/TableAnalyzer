#include "connected_components_finder.h"
#include <cassert>
#include <stack>

using std::stack;

vector<vector<int>> ConnectedComponentsFinder::Find() const
{
    const int numOfNodes = graph.size();
    
    vector<vector<int>> result;
    vector<bool> visited(numOfNodes, false);
    for(int source = 0; source < numOfNodes; ++source) {
        if(visited[source]) {
            continue;
        }
        const vector<int> component = findConnectedComponent(source, visited);
        assert(component.size() > 0);
        result.push_back(component);
    }
    return result;
}

vector<int> ConnectedComponentsFinder::findConnectedComponent(
    int source, vector<bool>& visited) const
{
    vector<int> result;

    stack<int> s;
    s.push(source);
    while(s.size() > 0) {
        const int u = s.top();
        s.pop();
        result.push_back(u);
        visited[u] = true;

        for(int v : graph[u]) {
            if(!visited[v]) {
                s.push(v);
            }
        }
    }

    return result;
}