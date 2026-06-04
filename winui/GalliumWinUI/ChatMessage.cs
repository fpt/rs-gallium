using Microsoft.UI.Xaml;
using Windows.UI;

namespace GalliumWinUI;

public sealed class ChatMessage
{
    public string Text { get; }
    public bool IsUser { get; }

    public HorizontalAlignment Alignment =>
        IsUser ? HorizontalAlignment.Right : HorizontalAlignment.Left;

    public Color BubbleColor =>
        IsUser
            ? Color.FromArgb(255, 0, 103, 192)  // blue
            : Color.FromArgb(255, 45, 45, 48);  // dark grey

    public ChatMessage(string text, bool isUser) { Text = text; IsUser = isUser; }
}
