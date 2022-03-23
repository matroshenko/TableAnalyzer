#include "cells_structure_builder.h"
#include "connected_components_finder.h"
#include <cassert>
#include <climits>
#include <algorithm>

using std::min;
using std::max;


CellsStructureBuilder::CellsStructureBuilder(
        const Tensor& _mergeRightMask, const Tensor& _mergeDownMask) :
    mergeRightMask(_mergeRightMask),
    mergeDownMask(_mergeDownMask)
{
    rowsCount = static_cast<int>(mergeRightMask.dim_size(0));
    colsCount = static_cast<int>(mergeDownMask.dim_size(1));

    assert(rowsCount > 0 && colsCount > 0);
}

vector<Rect> CellsStructureBuilder::Build() const
{
    vector<Rect> cells = buildInitialCells();
    mergeIntersectingCells(cells);
    return cells;
}

vector<Rect> CellsStructureBuilder::buildInitialCells() const
{
    const vector<vector<int>> graph = createGraph();
    const ConnectedComponentsFinder ccFinder(graph);
    const vector<vector<int>> components = ccFinder.Find();

    vector<Rect> result;
    for(const vector<int>& component : components) {
        result.push_back(getComponentRect(component));
    }
    return result;
}

vector<vector<int>> CellsStructureBuilder::createGraph() const
{
    const auto mergeRightMaskMatrix = mergeRightMask.matrix<bool>();
    const auto mergeDownMaskMatrix = mergeDownMask.matrix<bool>();

    vector<vector<int>> graph(rowsCount * colsCount);

    for(int i = 0; i < rowsCount; ++i) {
        for(int j = 0; j + 1 < colsCount; ++j) {
            if(mergeRightMaskMatrix(i, j)) {
                const int u = to1DIndex(i, j);
                const int v = to1DIndex(i, j+1);
                graph[u].push_back(v);
                graph[v].push_back(u);
            }
        }
    }

    for(int i = 0; i + 1 < rowsCount; ++i) {
        for(int j = 0; j < colsCount; ++j) {
            if(mergeDownMaskMatrix(i, j)) {
                const int u = to1DIndex(i, j);
                const int v = to1DIndex(i+1, j);
                graph[u].push_back(v);
                graph[v].push_back(u);
            }
        }
    }

    return graph;
}

pair<int, int> CellsStructureBuilder::to2DIndex(int index) const
{
    assert(0 <= index && index < rowsCount * colsCount);
    return {index / colsCount, index % colsCount};
}

int CellsStructureBuilder::to1DIndex(int rowIndex, int colIndex) const
{
    assert(0 <= rowIndex && rowIndex < rowsCount);
    assert(0 <= colIndex && colIndex < colsCount);

    return rowIndex * colsCount + colIndex;
}

Rect CellsStructureBuilder::getComponentRect(const vector<int>& component) const
{
    assert(component.size() > 0);

    int left = INT_MAX;
    int top = INT_MAX;
    int right = 0;
    int bottom = 0;
    for(int u : component) {
        const auto index2D = to2DIndex(u);
        top = min(top, index2D.first);
        bottom = max(bottom, index2D.first);
        left = min(left, index2D.second);
        right = max(right, index2D.second);
    }

    return Rect(left, top, right, bottom);
}

void CellsStructureBuilder::mergeIntersectingCells(vector<Rect>& cells) const
{
    int firstIndex = -1;
    int secondIndex = -1;
    while(findIntersectingCells(cells, firstIndex, secondIndex)) {
        assert(firstIndex < secondIndex);
        const Rect newCell = cells[firstIndex] | cells[secondIndex];
        cells.erase(cells.begin() + secondIndex);
        cells.erase(cells.begin() + firstIndex);
        cells.push_back(newCell);
    }
}

bool CellsStructureBuilder::findIntersectingCells(
    const vector<Rect>& cells, int& firstIndex, int& secondIndex) const
{
    const int n = cells.size();
    for(int i = 0; i + 1 < n; ++i) {
        const Rect& first = cells[i];
        for(int j = i + 1; j < n; ++j) {
            const Rect& second = cells[j];
            if(first.HasIntersection(second)) {
                firstIndex = i;
                secondIndex = j;
                return true;
            }
        }
    }
    return false;
}