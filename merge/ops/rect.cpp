#include "rect.h"
#include <cassert>
#include <algorithm>

using std::min;
using std::max;


Rect::Rect(int left, int top, int right, int bottom) :
    Left(left),
    Top(top),
    Right(right),
    Bottom(bottom)
{
    assert(Left <= Right && Top <= Bottom);
}

bool Rect::IsEmpty() const
{
    return Left == Right || Top == Bottom;
}

bool Rect::HasIntersection(const Rect& other) const
{
    return Left < other.Right && other.Left < Right
        && Top < other.Bottom && other.Top < Bottom;
}

Rect Rect::operator|(const Rect& other) const
{
    const int left = min(Left, other.Left);
    const int top = min(Top, other.Top);
    const int right = max(Right, other.Right);
    const int bottom = max(Bottom, other.Bottom);
    return Rect(left, top, right, bottom);
}