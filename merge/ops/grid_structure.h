#pragma once

#include <vector>

using std::vector;

struct Rect {
    int Left;
    int Top;
    int Right;
    int Bottom;

    Rect(int left, int top, int right, int bottom);

    bool IsEmpty() const;
    int Height() const { return Bottom - Top; }
    int Width() const { return Right - Left; }
    int GetArea() const { return Height() * Width(); }
};


class GridStructure {
public:
    GridStructure(const vector<int>& _horzPositions, const vector<int>& _vertPositions);

    int GetRowsCount() const { return horzPositions.size() - 1; }
    int GetColsCount() const { return vertPositions.size() - 1; }

    Rect GetCellRect(int rowIndex, int colIndex) const;

private:
    vector<int> horzPositions;
    vector<int> vertPositions;

    bool isSorted(const vector<int>& positions) const;
};