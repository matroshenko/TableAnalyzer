#pragma once

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