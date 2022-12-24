using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst.LowLevel;
using UnityEditor;
using System.Text.RegularExpressions;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

[assembly: InternalsVisibleTo("Unity.Burst.Editor.Tests")]

namespace Unity.Burst.Editor
{
    internal class BurstInspectorGUI : EditorWindow
    {
        private static bool Initialized;

        private static void EnsureInitialized()
        {
            if (Initialized)
            {
                return;
            }

            Initialized = true;

#if UNITY_2020_2_OR_NEWER
            BurstLoader.OnBurstShutdown += () =>
            {
                if (EditorWindow.HasOpenInstances<BurstInspectorGUI>())
                {
                    var window = EditorWindow.GetWindow<BurstInspectorGUI>("Burst Inspector");
                    window.Close();
                }
            };
#endif
        }

        private const string FontSizeIndexPref = "BurstInspectorFontSizeIndex";

        private static readonly string[] DisassemblyKindNames =
        {
            "Assembly",
            ".NET IL",
            "LLVM IR (Unoptimized)",
            "LLVM IR (Optimized)",
            "LLVM IR Optimisation Diagnostics"
        };

        private enum AssemblyOptions
        {
            PlainWithoutDebugInformation = 0,
            PlainWithDebugInformation = 1,
            EnhancedWithMinimalDebugInformation = 2,
            EnhancedWithFullDebugInformation = 3,
            ColouredWithMinimalDebugInformation = 4,
            ColouredWithFullDebugInformation = 5
        }
        private AssemblyOptions? _assemblyKind = null;
        private AssemblyOptions? _assemblyKindPrior = null;
        private AssemblyOptions _oldAssemblyKind;

        private bool SupportsEnhancedRendering => _disasmKind == DisassemblyKind.Asm || _disasmKind == DisassemblyKind.OptimizedIR || _disasmKind == DisassemblyKind.UnoptimizedIR;

        private static string[] DisasmOptions;

        private static string[] GetDisasmOptions()
        {
            if (DisasmOptions == null)
            {
                // We can't initialize this in BurstInspectorGUI.cctor because BurstCompilerOptions may not yet
                // have been initialized by BurstLoader. So we initialize on-demand here. This method doesn't need to
                // be thread-safe because it's only called from the UI thread.
                DisasmOptions = new[]
                {
                    "\n" + BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDump, NativeDumpFlags.Asm),
                    "\n" + BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDump, NativeDumpFlags.IL),
                    "\n" + BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDump, NativeDumpFlags.IR),
                    "\n" + BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDump, NativeDumpFlags.IROptimized),
                    "\n" + BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDump, NativeDumpFlags.IRPassAnalysis)
                };
            }
            return DisasmOptions;
        }

        private static readonly SplitterState TreeViewSplitterState = new SplitterState(new float[] { 30, 70 }, new int[] { 128, 128 }, null);

        private static readonly string[] TargetCpuNames = Enum.GetNames(typeof(BurstTargetCpu));

        private static readonly int[] FontSizes =
        {
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20
        };

        private static string[] _fontSizesText;
        internal const int _scrollbarThickness = 14;

        internal float _buttonOverlapInspectorView = 0;

        /// <remarks>Used because it's not legal to change layout of GUI in a frame without the users input.</remarks>
        private float widthtmp = -1;

        [NonSerialized]
        internal readonly BurstDisassembler _burstDisassembler;

        [SerializeField] private BurstTargetCpu _targetCpu = BurstTargetCpu.Auto;

        [SerializeField] private DisassemblyKind _disasmKind = DisassemblyKind.Asm;
        [SerializeField] private DisassemblyKind _oldDisasmKind = DisassemblyKind.Asm;

        [NonSerialized]
        private GUIStyle _fixedFontStyle;

        [NonSerialized]
        private int _fontSizeIndex = -1;

        [SerializeField] private int _previousTargetIndex = -1;

        [SerializeField] private bool _safetyChecks = false;
        [SerializeField] private bool _showBranchMarkers = true;
        [SerializeField] private bool _enhancedDisassembly = true;
        [SerializeField] private string _searchFilter;

        [SerializeField] private bool _sameTargetButDifferentAssemblyKind = false;
        [SerializeField] internal Vector2 _scrollPos;
        internal SearchField _searchFieldJobs;

        [SerializeField] private string _selectedItem;

        [NonSerialized]
        private List<BurstCompileTarget> _targets;

        [NonSerialized]
        internal LongTextArea _textArea;

        internal Rect _inspectorView;

        [NonSerialized]
        internal Font _font;

        [NonSerialized]
        internal BurstMethodTreeView _treeView;

        [NonSerialized]
        internal bool _initialized;

        [NonSerialized]
        private bool _requiresRepaint;

        private int FontSize => FontSizes[_fontSizeIndex];

        private readonly Regex _rx;

        private bool _leftClicked = false;

        [SerializeField] private bool _isCompileError = false;

        private enum KeyboardOperation
        {
            SelectAll,
            Copy,
            MoveLeft,
            MoveRight,
            MoveUp,
            MoveDown,
        }

        private Dictionary<Event, KeyboardOperation> _keyboardEvents;

        private void FillKeyboardEvent()
        {
            if (_keyboardEvents != null)
            {
                return;
            }

            _keyboardEvents = new Dictionary<Event, KeyboardOperation>
            {
                { Event.KeyboardEvent("^a"), KeyboardOperation.SelectAll },
                { Event.KeyboardEvent("^c"), KeyboardOperation.Copy },
                { Event.KeyboardEvent("#left"), KeyboardOperation.MoveLeft },
                { Event.KeyboardEvent("#right"), KeyboardOperation.MoveRight },
                { Event.KeyboardEvent("#down"), KeyboardOperation.MoveDown },
                { Event.KeyboardEvent("#up"), KeyboardOperation.MoveUp }
            };
        }

        public BurstInspectorGUI()
        {
            _burstDisassembler = new BurstDisassembler();
            var pattern = @"^\(\d+,\d+\):\sBurst\serror";
            _rx = new Regex(pattern);
        }

        private bool DisplayAssemblyKind(Enum assemblyKind)
        {
            AssemblyOptions assemblyKind1 = (AssemblyOptions)assemblyKind;
            if (_disasmKind != DisassemblyKind.Asm || _isCompileError)
            {
                return assemblyKind1 == AssemblyOptions.PlainWithoutDebugInformation;
            }
            return true;
        }

        public void OnEnable()
        {
            EnsureInitialized();

            if (_treeView == null) _treeView = new BurstMethodTreeView(new TreeViewState(), () => _searchFilter);

            if (_keyboardEvents == null) FillKeyboardEvent();

            var assemblyList = BurstReflection.EditorAssembliesThatCanPossiblyContainJobs;

            Task.Run(
                () =>
                {
                    // Do this stuff asynchronously.
                    var result = BurstReflection.FindExecuteMethods(assemblyList, BurstReflectionAssemblyOptions.None);
                    _targets = result.CompileTargets;
                    _targets.Sort((left, right) => string.Compare(left.GetDisplayName(), right.GetDisplayName(), StringComparison.Ordinal));
                    return result;
                })
                .ContinueWith(t =>
                {
                    // Do this stuff on the main (UI) thread.
                    if (t.Status == TaskStatus.RanToCompletion)
                    {
                        foreach (var logMessage in t.Result.LogMessages)
                        {
                            switch (logMessage.LogType)
                            {
                                case BurstReflection.LogType.Warning:
                                    Debug.LogWarning(logMessage.Message);
                                    break;
                                case BurstReflection.LogType.Exception:
                                    Debug.LogException(logMessage.Exception);
                                    break;
                                default:
                                    throw new InvalidOperationException();
                            }
                        }

                        _treeView.Targets = _targets;
                        _treeView.Reload();

                        if (_selectedItem == null || !_treeView.TrySelectByDisplayName(_selectedItem))
                        {
                            _previousTargetIndex = -1;
                            _scrollPos = Vector2.zero;
                        }

                        _requiresRepaint = true;
                        _initialized = true;
                    }
                    else if (t.Exception != null)
                    {
                        Debug.LogError($"Could not load Inspector: {t.Exception}");
                    }
                });
        }

        private void CleanupFont()
        {
            if (_font != null)
            {
                DestroyImmediate(_font, true);
                _font = null;
            }
        }

        public void OnDisable()
        {
            CleanupFont();
        }

        public void Update()
        {
            // Need to do this because if we call Repaint from anywhere else,
            // it doesn't do anything if this window is not currently focused.
            if (_requiresRepaint)
            {
                Repaint();
                _requiresRepaint = false;
            }
        }

        /// <summary>
        /// Checks if there is space for given content withs style, and starts new horizontalgroup
        /// if there is no space on this line.
        /// </summary>
        private void FlowToNewLine(ref float remainingWidth, float width, GUIStyle style, GUIContent content)
        {
            Vector2 size = style.CalcSize(content);
            float sizeX = size.x + _scrollbarThickness / 2;
            if (sizeX >= remainingWidth)
            {
                _buttonOverlapInspectorView += size.y + 2;
                remainingWidth = width - sizeX;
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
            }
            else
            {
                remainingWidth -= sizeX;
            }
        }

        private bool IsRaw(AssemblyOptions kind)
        {
            return kind == AssemblyOptions.PlainWithoutDebugInformation || kind == AssemblyOptions.PlainWithDebugInformation;
        }

        private bool IsEnhanced(AssemblyOptions kind)
        {
            return !IsRaw(kind);
        }

        private bool IsColoured(AssemblyOptions kind)
        {
            return kind == AssemblyOptions.ColouredWithMinimalDebugInformation || kind == AssemblyOptions.ColouredWithFullDebugInformation;
        }

        /// <summary>
        /// Renders buttons bar, and handles saving/loading of _assemblyKind options when changing in inspector settings
        /// that disable/enables some options for _assemblyKind.
        /// </summary>
        private void HandleButtonBars(BurstCompileTarget target, bool targetChanged, out int fontIndex, out bool collapse, out bool focusCode)
        {
            var prevWasCompileError = _isCompileError;
            _isCompileError = _rx.IsMatch(target.RawDisassembly ?? "");

            // Refresh if any options are changed
            // TODO: make given width not depend on a guess for the real width.

            // We can only make an educated guess for the correct width.
            if (widthtmp == -1)
            {
                widthtmp = (position.width * 2) / 3 - _scrollbarThickness;
            }

            RenderButtonBars(widthtmp, target, out fontIndex, out collapse, out focusCode);

            var disasmKindChanged = _oldDisasmKind != _disasmKind;

            // Handles saving and loading _assemblyKind option when going between settings, that disable/enable some options for it
            if ((disasmKindChanged && _oldDisasmKind == DisassemblyKind.Asm && !_isCompileError)
                || (targetChanged && !prevWasCompileError && _isCompileError && _disasmKind == DisassemblyKind.Asm))
            {
                // save when _disasmKind changed from Asm WHEN we are not looking at a burst compile error,
                // or when target changed from non compile error to compile error and current _disasmKind is Asm.
                _oldAssemblyKind = (AssemblyOptions)_assemblyKind;
            }
            else if ((disasmKindChanged && _disasmKind == DisassemblyKind.Asm && !_isCompileError) ||
                (targetChanged && prevWasCompileError && _disasmKind == DisassemblyKind.Asm))
            {
                // load when _diasmKind changed to Asm and we are not at burst compile error,
                // or when target changed from a burst compile error while _disasmKind is Asm.
                _assemblyKind = _oldAssemblyKind;
            }

            // if _assemblyKind is something that is not available, force it up to PlainWithoutDebugInformation.
            if ((_disasmKind != DisassemblyKind.Asm && _assemblyKind != AssemblyOptions.PlainWithoutDebugInformation)
            || _isCompileError)
            {
                _assemblyKind = AssemblyOptions.PlainWithoutDebugInformation;
            }
        }

        private void RenderButtonBars(float width, BurstCompileTarget target, out int fontIndex, out bool collapse, out bool focus)
        {
            float remainingWidth = width;

            var contentDisasm = new GUIContent("Enhanced With Minimal Debug Information");
            var contentSafety = new GUIContent("Safety Checks");
            var sizeContentAssemply = new GUIContent("ARMV8A_AARCH64_HALFFP"); // Only used for proper width of popup.
            var contentLabelFontSize = new GUIContent("Font Size");
            var sizeContentFontSize = new GUIContent("9999");                  // Only used for proper width of popup.
            var contentCollapseToCode = new GUIContent("Focus on Code");
            var contentExpandAll = new GUIContent("Expand All");
            var contentBranchLines = new GUIContent("Show Branch Flow");

            GUILayout.BeginHorizontal();

            // We are forced to hand over a GUIConttent for EnumPopup.
            // use contentDisasm here for it to use a proper size in FlowToNewLine.
            FlowToNewLine(ref remainingWidth, width, EditorStyles.popup, contentDisasm);
            _assemblyKind = (AssemblyOptions)EditorGUILayout.EnumPopup(GUIContent.none, _assemblyKind, DisplayAssemblyKind, true);

            var contentSafetySize = EditorStyles.toggle.CalcSize(contentSafety).x;
            FlowToNewLine(ref remainingWidth, width, EditorStyles.toggle, contentSafety);
            _safetyChecks = GUILayout.Toggle(_safetyChecks, contentSafety, EditorStyles.toggle, GUILayout.MaxWidth(contentSafetySize));

            EditorGUI.BeginDisabledGroup(!target.HasRequiredBurstCompileAttributes);

            FlowToNewLine(ref remainingWidth, width, EditorStyles.popup, sizeContentAssemply);
            _targetCpu = (BurstTargetCpu)EditorGUILayout.Popup((int)_targetCpu, TargetCpuNames, EditorStyles.popup);

            FlowToNewLine(ref remainingWidth, width, EditorStyles.label, contentLabelFontSize);
            FlowToNewLine(ref remainingWidth, width - EditorStyles.label.CalcSize(contentLabelFontSize).x - _scrollbarThickness / 2, EditorStyles.popup, sizeContentFontSize);
            GUILayout.Label(contentLabelFontSize);
            fontIndex = EditorGUILayout.Popup(_fontSizeIndex, _fontSizesText, EditorStyles.popup);

            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginDisabledGroup(!IsEnhanced((AssemblyOptions)_assemblyKind) || !SupportsEnhancedRendering || _isCompileError);

            FlowToNewLine(ref remainingWidth, width, EditorStyles.miniButton, contentCollapseToCode);
            focus = GUILayout.Button(contentCollapseToCode, EditorStyles.miniButton);

            FlowToNewLine(ref remainingWidth, width, EditorStyles.miniButton, contentExpandAll);
            collapse = GUILayout.Button(contentExpandAll, EditorStyles.miniButton);

            // Want to avoid the toggle expanding it's click-area beyond text, when given its own line.
            var contentBranchLinesSize = EditorStyles.toggle.CalcSize(contentBranchLines).x;
            FlowToNewLine(ref remainingWidth, width, EditorStyles.toggle, contentBranchLines);
            _showBranchMarkers = GUILayout.Toggle(_showBranchMarkers, contentBranchLines, EditorStyles.toggle, GUILayout.MaxWidth(contentBranchLinesSize));

            EditorGUI.EndDisabledGroup();

            GUILayout.EndHorizontal();

            _oldDisasmKind = _disasmKind;
            _disasmKind = (DisassemblyKind)GUILayout.Toolbar((int)_disasmKind, DisassemblyKindNames, GUILayout.ExpandWidth(true), GUILayout.MinWidth(5 * 10));
        }

        /// <summary>
        /// Handles mouse events for selecting text.
        /// </summary>
        /// <remarks>
        /// Must be called after Render(...), as it uses the mouse events, and Render(...)
        /// need mouse events for buttons etc.
        /// </remarks>
        private void HandleMouseEventForSelection(Rect workingArea, int controlID)
        {
            var evt = Event.current;
            var mousePos = evt.mousePosition;

            if (_textArea.MouseOutsideView(workingArea, mousePos, controlID))
            {
                return;
            }

            switch (evt.type)
            {
                case EventType.MouseDown:
                    // button 0 is left and 1 is right
                    if (evt.button == 0)
                    {
                        _textArea.MouseClicked(evt.shift, mousePos, controlID);
                    }
                    else
                    {
                        _leftClicked = true;
                    }
                    evt.Use();
                    break;
                case EventType.MouseDrag:
                    _textArea.DragMouse(mousePos);
                    evt.Use();
                    break;
                case EventType.MouseUp:
                    _textArea.MouseReleased();
                    evt.Use();
                    break;
                case EventType.ScrollWheel:
                    _textArea.DoScroll(evt.delta.y);
                    // we cannot Use() (consume) scrollWheel events, as they are still needed in EndScrollView.
                    break;
            }
        }

        /// <remarks>
        /// Must be called after Render(...) because of depenency on LongTextArea.finalAreaSize.
        /// </remarks>
        private void HandleKeyboardEventForSelection(Rect workingArea, bool showBranchMarkers)
        {
            var evt = Event.current;

            if (!_keyboardEvents.TryGetValue(evt, out var op))
            {
                return;
            }

            switch (op)
            {
                case KeyboardOperation.SelectAll:
                    _textArea.SelectAll();
                    evt.Use();
                    break;
                case KeyboardOperation.Copy:
                    _textArea.DoSelectionCopy();
                    evt.Use();
                    break;
                case KeyboardOperation.MoveLeft:
                    _textArea.MoveSelectionLeft(workingArea, showBranchMarkers);
                    evt.Use();
                    break;
                case KeyboardOperation.MoveRight:
                    _textArea.MoveSelectionRight(workingArea, showBranchMarkers);
                    evt.Use();
                    break;
                case KeyboardOperation.MoveUp:
                    _textArea.MoveSelectionUp(workingArea);
                    evt.Use();
                    break;
                case KeyboardOperation.MoveDown:
                    _textArea.MoveSelectionDown(workingArea);
                    evt.Use();
                    break;
            }
        }

        public void OnGUI()
        {
            if (!_initialized)
            {
                GUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                GUILayout.BeginVertical();
                GUILayout.FlexibleSpace();
                GUILayout.Label("Loading...");
                GUILayout.FlexibleSpace();
                GUILayout.EndVertical();
                GUILayout.FlexibleSpace();
                GUILayout.EndHorizontal();
                return;
            }
            // used to give hot control to inspector when a mouseDown event has happened.
            // This way we can register a mouseUp happening outside inspector.
            int controlID = GUIUtility.GetControlID(FocusType.Passive);

            // Make sure that editor options are synchronized
            BurstEditorOptions.EnsureSynchronized();

            if (_fontSizesText == null)
            {
                _fontSizesText = new string[FontSizes.Length];
                for (var i = 0; i < FontSizes.Length; ++i) _fontSizesText[i] = FontSizes[i].ToString();
            }

            if (_fontSizeIndex == -1)
            {
                _fontSizeIndex = EditorPrefs.GetInt(FontSizeIndexPref, 5);
                _fontSizeIndex = Math.Max(0, _fontSizeIndex);
                _fontSizeIndex = Math.Min(_fontSizeIndex, FontSizes.Length - 1);
            }

            if (_fixedFontStyle == null || _fixedFontStyle.font == null) // also check .font as it's reset somewhere when going out of play mode.
            {
                _fixedFontStyle = new GUIStyle(GUI.skin.label);
                string fontName;
                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    fontName = "Consolas";
                }
                else
                {
                    fontName = "Courier";
                }

                CleanupFont();

                _font = Font.CreateDynamicFontFromOSFont(fontName, FontSize);
                _fixedFontStyle.font = _font;
                _fixedFontStyle.fontSize = FontSize;
            }

            if (_searchFieldJobs == null) _searchFieldJobs = new SearchField();

            if (_textArea == null) _textArea = new LongTextArea();

            GUILayout.BeginHorizontal();

            // SplitterGUILayout.BeginHorizontalSplit is internal in Unity but we don't have much choice
            SplitterGUILayout.BeginHorizontalSplit(TreeViewSplitterState);

            GUILayout.BeginVertical(GUILayout.Width(position.width / 3));

            GUILayout.Label("Compile Targets", EditorStyles.boldLabel);

            var newFilter = _searchFieldJobs.OnGUI(_searchFilter);

            if (newFilter != _searchFilter)
            {
                _searchFilter = newFilter;
                _treeView.Reload();
            }

            // Does not give proper rect during layout event.
            _inspectorView = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.ExpandHeight(true), GUILayout.ExpandWidth(true));

            _treeView.OnGUI(_inspectorView);

            GUILayout.EndVertical();

            GUILayout.BeginVertical();

            var selection = _treeView.GetSelection();
            if (selection.Count == 1)
            {
                var targetIndex = selection[0];
                var target = _targets[targetIndex - 1];
                var targetOptions = target.Options;

                bool targetChanged = _previousTargetIndex != targetIndex;

                _previousTargetIndex = targetIndex;

                // Stash selected item name to handle domain reloads more gracefully
                _selectedItem = target.GetDisplayName();

                if (_assemblyKind == null)
                {
                    if (_enhancedDisassembly)
                    {
                        _assemblyKind = AssemblyOptions.ColouredWithMinimalDebugInformation;
                    }
                    else
                    {
                        _assemblyKind = AssemblyOptions.PlainWithoutDebugInformation;
                    }
                    _oldAssemblyKind = (AssemblyOptions)_assemblyKind;
                }

                // We are currently formatting only Asm output
                var isTextFormatted = IsEnhanced((AssemblyOptions)_assemblyKind) && SupportsEnhancedRendering;

                // Depending if we are formatted or not, we don't render the same text
                var textToRender = target.RawDisassembly?.TrimStart('\n');

                // Only refresh if we are switching to a new selection that hasn't been disassembled yet
                // Or we are changing disassembly settings (safety checks / enhanced disassembly)
                var targetRefresh = textToRender == null
                                    || target.DisassemblyKind != _disasmKind
                                    || targetOptions.EnableBurstSafetyChecks != _safetyChecks
                                    || target.TargetCpu != _targetCpu
                                    || target.IsDarkMode != EditorGUIUtility.isProSkin;

                // Used for _textArea.SetText(...) to distinguished when to call SetDisassembler(...).
                var isAssemblyKindJustChanged = false;

                if (_assemblyKindPrior != _assemblyKind)
                {
                    targetRefresh = true;
                    _assemblyKindPrior = _assemblyKind;  // Needs to be refreshed, as we need to change disassembly options

                    // If the target did not changed but our assembly kind did, we need to remember this.
                    if (!targetChanged)
                    {
                        _sameTargetButDifferentAssemblyKind = true;
                        isAssemblyKindJustChanged = true;
                    }
                }

                // If the previous target changed the assembly kind and we have a target change, we need to
                // refresh the assembly because we'll have cached the previous assembly kinds output rather
                // than the one requested.
                if (_sameTargetButDifferentAssemblyKind && targetChanged)
                {
                    targetRefresh = true;
                    _sameTargetButDifferentAssemblyKind = false;
                }

                if (targetRefresh)
                {
                    var options = new StringBuilder();

                    target.TargetCpu = _targetCpu;
                    target.DisassemblyKind = _disasmKind;
                    targetOptions.EnableBurstSafetyChecks = _safetyChecks;
                    target.IsDarkMode = EditorGUIUtility.isProSkin;
                    targetOptions.EnableBurstCompileSynchronously = true;

                    string defaultOptions;
                    if (targetOptions.TryGetOptions(target.IsStaticMethod ? (MemberInfo)target.Method : target.JobType, true, out defaultOptions))
                    {
                        options.AppendLine(defaultOptions);

                        // Disables the 2 current warnings generated from code (since they clutter up the inspector display)
                        // BC1370 - throw inside code not guarded with ConditionalSafetyCheck attribute
                        // BC1322 - loop intrinsic on loop that has been optimised away
                        options.AppendLine($"{BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDisableWarnings, "BC1370;BC1322")}");

                        options.AppendLine($"{BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionTarget, TargetCpuNames[(int)_targetCpu])}");

                        switch (_assemblyKind)
                        {
                            case AssemblyOptions.EnhancedWithMinimalDebugInformation:
                            case AssemblyOptions.ColouredWithMinimalDebugInformation:
                                options.AppendLine($"{BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDebug, "2")}");
                                break;
                            case AssemblyOptions.ColouredWithFullDebugInformation:
                            case AssemblyOptions.EnhancedWithFullDebugInformation:
                            case AssemblyOptions.PlainWithDebugInformation:
                                options.AppendLine($"{BurstCompilerOptions.GetOption(BurstCompilerOptions.OptionDebug, "1")}");
                                break;
                            default:
                            case AssemblyOptions.PlainWithoutDebugInformation:
                                break;
                        }

                        var baseOptions = options.ToString();

                        target.RawDisassembly = GetDisassembly(target.Method, baseOptions + GetDisasmOptions()[(int)_disasmKind]);

                        target.FormattedDisassembly = null;
                        textToRender = target.RawDisassembly.TrimStart('\n');
                    }
                }

                _buttonOverlapInspectorView = 0;
                HandleButtonBars(target, targetChanged, out var fontSize, out var expandAllBlocks, out var focusCode);

                if (textToRender != null)
                {
					      // we should only call SetDisassembler(...) the first time assemblyKind is changed with same target.
            		// Otherwise it will kep re-initializing fields such as _folded, meaning we can no longer fold/unfold.

                    if (!_textArea.IsTextSet(textToRender) || isAssemblyKindJustChanged)
                    {
                        _textArea.SetText(textToRender, target.IsDarkMode, isAssemblyKindJustChanged, _burstDisassembler, isTextFormatted && _burstDisassembler.Initialize(
                            textToRender, FetchAsmKind(_targetCpu, _disasmKind), target.IsDarkMode, IsColoured((AssemblyOptions)_assemblyKind)));
                    }
                    if (targetChanged)
                    {
                        _scrollPos = Vector2.zero;
                    }

                    // Might want to feed it styles for scrollbars to fix them going too small!

                    _scrollPos = GUILayout.BeginScrollView(_scrollPos, true, true);

                    // TODO: This code is needed for hovering over instructions
                    //if (Event.current.type == EventType.Repaint)    // we always want mouse position feedback
                    //{
                    //    var mousePos = Event.current.mousePosition;
                    //    _textArea.Interact(mousePos);
                    //}

                    // Fixing lastRectSize to actually be size of scroll view
                    _inspectorView.position = _scrollPos;
                    _inspectorView.width = position.width - (_inspectorView.width + _scrollbarThickness);
                    _inspectorView.height -= (_scrollbarThickness + 4 + _buttonOverlapInspectorView); //+4 for alignment.

                    // repaint indicate end of frame, so we can alter width for menu items to new correct.
                    if (Event.current.type == EventType.Repaint)
                    {
                        widthtmp = _inspectorView.width - 3*_scrollbarThickness; // 3* because of magical gnomes
                    }

                    _textArea.Render(_fixedFontStyle, _inspectorView, _showBranchMarkers);

                    if (Event.current.type != EventType.Layout)
                    {
                        HandleMouseEventForSelection(_inspectorView, controlID);
                        HandleKeyboardEventForSelection(_inspectorView, _showBranchMarkers);
                    }

                    if (_leftClicked)
                    {
                        GenericMenu menu = new GenericMenu();

                        menu.AddItem(EditorGUIUtility.TrTextContent("Copy Selection"), false, _textArea.DoSelectionCopy);
                        menu.AddItem(EditorGUIUtility.TrTextContent("Select All"), false, _textArea.SelectAll);
                        menu.ShowAsContext();

                        _leftClicked = false;
                    }

                    GUILayout.EndScrollView();
                }

                if (fontSize != _fontSizeIndex)
                {
                    _textArea.Invalidate();
                    _fontSizeIndex = fontSize;
                    EditorPrefs.SetInt(FontSizeIndexPref, fontSize);
                    _fixedFontStyle = null;
                }

                if (expandAllBlocks)
                {
                    _textArea.ExpandAllBlocks();
                }

                if (focusCode)
                {
                    _textArea.FocusCodeBlocks();
                }
            }

            GUILayout.EndVertical();

            SplitterGUILayout.EndHorizontalSplit();

            GUILayout.EndHorizontal();
        }

        private static string GetDisassembly(MethodInfo method, string options)
        {
            try
            {
                var result = BurstCompilerService.GetDisassembly(method, options);
                if (result.IndexOf('\t') >= 0)
                {
                    result = result.Replace("\t", "        ");
                }

                // Workaround to remove timings
                if (result.Contains("Burst timings"))
                {
                    var index = result.IndexOf("While compiling", StringComparison.Ordinal);
                    if (index > 0)
                    {
                        result = result.Substring(index);
                    }
                }

                return result;
            }
            catch (Exception e)
            {
                return "Failed to compile:\n" + e.Message;
            }
        }

        private static BurstDisassembler.AsmKind FetchAsmKind(BurstTargetCpu cpu, DisassemblyKind kind)
        {
            if (kind == DisassemblyKind.Asm)
            {
                switch (cpu)
                {
                    case BurstTargetCpu.ARMV7A_NEON32:
                    case BurstTargetCpu.ARMV8A_AARCH64:
                    case BurstTargetCpu.ARMV8A_AARCH64_HALFFP:
                    case BurstTargetCpu.THUMB2_NEON32:
                        return BurstDisassembler.AsmKind.ARM;
                    case BurstTargetCpu.WASM32:
                        return BurstDisassembler.AsmKind.Wasm;
                }
                return BurstDisassembler.AsmKind.Intel;
            }
            else
            {
                return BurstDisassembler.AsmKind.LLVMIR;
            }
        }
    }

    internal class BurstMethodTreeView : TreeView
    {
        private readonly Func<string> _getFilter;

        public BurstMethodTreeView(TreeViewState state, Func<string> getFilter) : base(state)
        {
            _getFilter = getFilter;
            showBorder = true;
        }

        public List<BurstCompileTarget> Targets { get; set; }

        protected override TreeViewItem BuildRoot()
        {
            var root = new TreeViewItem {id = 0, depth = -1, displayName = "Root"};
            var allItems = new List<TreeViewItem>();

            if (Targets != null)
            {
                allItems.Capacity = Targets.Count;
                var id = 1;
                var filter = _getFilter();
                foreach (var t in Targets)
                {
                    var displayName = t.GetDisplayName();
                    if (string.IsNullOrEmpty(filter) || displayName.IndexOf(filter, 0, displayName.Length, StringComparison.InvariantCultureIgnoreCase) >= 0)
                    {
                        allItems.Add(new TreeViewItem { id = id, depth = 0, displayName = displayName });
                    }

                    ++id;
                }
            }

            SetupParentsAndChildrenFromDepths(root, allItems);

            return root;
        }

        internal bool TrySelectByDisplayName(string name)
        {
            var id = 1;
            foreach (var t in Targets)
            {
                if (t.GetDisplayName() == name)
                {
                    try
                    {
                        SetSelection(new[] { id });
                        FrameItem(id);
                        return true;
                    }
                    catch (ArgumentException)
                    {
                        // When a search is made in the job list, such that the job we search for is filtered away
                        // FrameItem(id) will throw a dictionary error. So we catch this, and tell the caller that
                        // it cannot be selected.
                        return false;
                    }
                }
                else
                {
                    ++id;
                }
            }
            return false;
        }

        protected override void RowGUI(RowGUIArgs args)
        {
            var target = Targets[args.item.id - 1];
            var wasEnabled = GUI.enabled;
            GUI.enabled = target.HasRequiredBurstCompileAttributes;
            base.RowGUI(args);
            GUI.enabled = wasEnabled;
        }

        protected override bool CanMultiSelect(TreeViewItem item)
        {
            return false;
        }
    }
}
