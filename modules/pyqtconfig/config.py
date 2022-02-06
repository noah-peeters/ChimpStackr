# -*- coding: utf-8 -*-
""" PyQtConfig is a simple API for handling, persisting and synchronising
    configuration within PyQt applications.
"""
import logging

import types
from collections import OrderedDict

# Import PyQt5/PySide2 classes
from .qt import (
    QComboBox,
    QCheckBox,
    QAction,
    QActionGroup,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QLineEdit,
    QListWidget,
    QSlider,
    QButtonGroup,
    QTabWidget,
    QVariant,
    Qt,
    QMutex,
    QMutexLocker,
    QSettings,
    QObject,
    Signal,
    QApplication,
)
from .qt import QtWidgets

try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et

import warnings
import json
import os
import pathlib
from math import ceil
import sys

RECALCULATE_ALL = 1
RECALCULATE_VIEW = 2


def types_MethodType(fn, handler):
    try:
        return types.MethodType(fn, handler, type(handler))
    except TypeError:
        return types.MethodType(fn, handler)


def _convert_list_type_from_XML(vs):
    """
    Lists are a complex type with possibility for mixed sub-types. Therefore
    each sub-entity must be wrapped with a type specifier.
    """
    # ConfigListItem is legacy
    vlist = vs.findall("ListItem") + vs.findall("ConfigListItem")
    l = []
    for xconfig in vlist:
        v = xconfig.text
        if xconfig.get("type") in CONVERT_TYPE_FROM_XML:
            # Recursive; woo!
            v = CONVERT_TYPE_FROM_XML[xconfig.get("type")](xconfig)
        l.append(v)
    return l


def _convert_list_type_to_XML(co, vs):
    """
    Lists are a complex type with possibility for mixed sub-types. Therefore
    each sub-entity must be wrapped with a type specifier.
    """
    for cv in vs:
        c = et.SubElement(co, "ListItem")
        t = type(cv).__name__
        c.set("type", t)
        c = CONVERT_TYPE_TO_XML[t](c, cv)
    return co


def _convert_dict_type_from_XML(vs):
    """
    Dicts are a complex type with possibility for mixed sub-types. Therefore
    each sub-entity must be wrapped with a type specifier.
    """
    vlist = vs.findall("DictItem")
    d = {}
    for xconfig in vlist:
        v = xconfig.text
        if xconfig.get("type") in CONVERT_TYPE_FROM_XML:
            # Recursive; woo!
            v = CONVERT_TYPE_FROM_XML[xconfig.get("type")](xconfig)
        d[xconfig.get("key")] = v
    return d


def _convert_dict_type_to_XML(co, vs):
    """
    Dicts are a complex type with possibility for mixed sub-types. Therefore
    each sub-entity must be wrapped with a type specifier.
    """
    for k, v in vs.items():
        c = et.SubElement(co, "DictItem")
        t = type(v).__name__
        c.set("type", t)
        c.set("key", k)
        c = CONVERT_TYPE_TO_XML[t](c, v)
    return co


def _apply_text_str(co, s):
    co.text = str(s)
    return co


CONVERT_TYPE_TO_XML = {
    "str": _apply_text_str,
    "unicode": _apply_text_str,
    "int": _apply_text_str,
    "float": _apply_text_str,
    "bool": _apply_text_str,
    "list": _convert_list_type_to_XML,
    "tuple": _convert_list_type_to_XML,
    "dict": _convert_dict_type_to_XML,
    "NoneType": _apply_text_str,
}

CONVERT_TYPE_FROM_XML = {
    "str": lambda x: str(x.text),
    "unicode": lambda x: str(x.text),
    "int": lambda x: int(x.text),
    "float": lambda x: float(x.text),
    "bool": lambda x: bool(x.text.lower() == "true"),
    "list": _convert_list_type_from_XML,
    "tuple": _convert_list_type_from_XML,
    "dict": _convert_dict_type_from_XML,
    "NoneType": lambda x: None,
}


def build_dict_mapper(mdict):
    """
    Build a map function pair for forward and reverse mapping from a specified
    dict

    Mapping requires both a forward and reverse (get, set) mapping function.
    This function is used to automatically convert a supplied dict to a forward
    and reverse paired lambda.

    :param mdict: A dictionary of display values (keys) and stored values
                  (values)
    :type mdict: dict
    :rtype: 2-tuple of lambdas that perform forward and reverse map

    """
    rdict = {v: k for k, v in mdict.items()}
    return (
        lambda x: mdict[x] if x in mdict else x,
        lambda x: rdict[x] if x in rdict else x,
    )


try:
    # Python2.7
    unicode
except NameError:
    # Python3 recoding
    def unicode(s):
        if isinstance(s, bytes):
            return s.decode("utf-8")
        return s


# Basestring for typechecking
try:
    basestring
except NameError:
    basestring = str


def build_tuple_mapper(mlist):
    """
    Build a map function pair for forward and reverse mapping from a specified
    list of tuples

    :param mlist: A list of tuples of display values (keys) and stored values
                  (values)
    :type mlist: list-of-tuples
    :rtype: 2-tuple of lambdas that perform forward and reverse map

    """
    rdict = {v: k for k, v in mlist}
    return (
        lambda x: mlist[x] if x in mlist else x,
        lambda x: rdict[x] if x in rdict else x,
    )


# CUSTOM HANDLERS

# QComboBox
def _get_QComboBox(self):
    """
    Get value QCombobox via re-mapping filter
    """
    return self._get_map(self.currentText())


def _set_QComboBox(self, v):
    """
    Set value QCombobox via re-mapping filter
    """
    self.setCurrentIndex(self.findText(unicode(self._set_map(v))))


def _event_QComboBox(self):
    """
    Return QCombobox change event signal
    """
    return self.currentIndexChanged


# QCheckBox
def _get_QCheckBox(self):
    """
    Get state of QCheckbox
    """
    return self.isChecked()


def _set_QCheckBox(self, v):
    """
    Set state of QCheckbox
    """
    self.setChecked(v)


def _event_QCheckBox(self):
    """
    Return state change signal for QCheckbox
    """
    return self.stateChanged


# QAction
def _get_QAction(self):
    """
    Get checked state of QAction
    """
    return self.isChecked()


def _set_QAction(self, v):
    """
    Set checked state of QAction
    """
    self.setChecked(v)


def _event_QAction(self):
    """
    Return state change signal for QAction
    """
    return self.toggled


# QActionGroup
def _get_QActionGroup(self):
    """
    Get checked state of QAction
    """
    if self.checkedAction():
        return self.actions().index(self.checkedAction())
    return None


def _set_QActionGroup(self, v):
    """
    Set checked state of QAction
    """
    self.actions()[v].setChecked(True)


def _event_QActionGroup(self):
    """
    Return state change signal for QAction
    """
    return self.triggered


# QPushButton
def _get_QPushButton(self):
    """
    Get checked state of QPushButton
    """
    return self.isChecked()


def _set_QPushButton(self, v):
    """
    Set checked state of QPushButton
    """
    self.setChecked(v)


def _event_QPushButton(self):
    """
    Return state change signal for QPushButton
    """
    return self.toggled


# QSpinBox
def _get_QSpinBox(self):
    """
    Get current value for QSpinBox
    """
    return self.value()


def _set_QSpinBox(self, v):
    """
    Set current value for QSpinBox
    """
    self.setValue(v)


def _event_QSpinBox(self):
    """
    Return value change signal for QSpinBox
    """
    return self.valueChanged


# QDoubleSpinBox
def _get_QDoubleSpinBox(self):
    """
    Get current value for QDoubleSpinBox
    """
    return self.value()


def _set_QDoubleSpinBox(self, v):
    """
    Set current value for QDoubleSpinBox
    """
    self.setValue(v)


def _event_QDoubleSpinBox(self):
    """
    Return value change signal for QDoubleSpinBox
    """
    return self.valueChanged


# QPlainTextEdit
def _get_QPlainTextEdit(self):
    """
    Get current document text for QPlainTextEdit
    """
    return self.document().toPlainText()


def _set_QPlainTextEdit(self, v):
    """
    Set current document text for QPlainTextEdit
    """
    self.setPlainText(unicode(v))


def _event_QPlainTextEdit(self):
    """
    Return current value changed signal for QPlainTextEdit box.

    Note that this is not a native Qt signal but a signal manually fired on
    the user's pressing the "Apply changes" to the code button. Attaching
    to the modified signal would trigger recalculation on every key-press.
    """
    return self.textChanged


# QLineEdit
def _get_QLineEdit(self):
    """
    Get current text for QLineEdit
    """
    return self._get_map(self.text())


def _set_QLineEdit(self, v):
    """
    Set current text for QLineEdit
    """
    self.setText(unicode(self._set_map(v)))


def _event_QLineEdit(self):
    """
    Return current value changed signal for QLineEdit box.
    """
    return self.textChanged


# CodeEditor
def _get_CodeEditor(self):
    """
    Get current document text for CodeEditor. Wraps _get_QPlainTextEdit.
    """
    _get_QPlainTextEdit(self)


def _set_CodeEditor(self, v):
    """
    Set current document text for CodeEditor. Wraps _set_QPlainTextEdit.
    """
    _set_QPlainTextEdit(self, unicode(v))


def _event_CodeEditor(self):
    """
    Return current value changed signal for
    CodeEditor box. Wraps _event_QPlainTextEdit.
    """
    return _event_QPlainTextEdit(self)


# QListWidget
def _get_QListWidget(self):
    """
    Get currently selected values in QListWidget via re-mapping filter.

    Selected values are returned as a list.
    """
    return [self._get_map(s.text()) for s in self.selectedItems()]


def _set_QListWidget(self, v):
    """
    Set currently selected values in QListWidget via re-mapping filter.

    Supply values to be selected as a list.
    """
    if v:
        for s in v:
            self.findItems(unicode(self._set_map(s)), Qt.MatchExactly)[0].setSelected(
                True
            )


def _event_QListWidget(self):
    """
    Return current selection changed signal for QListWidget.
    """
    return self.itemSelectionChanged


# QListWidgetWithAddRemoveEvent
def _get_QListWidgetAddRemove(self):
    """
    Get current values in QListWidget via re-mapping filter.

    Selected values are returned as a list.
    """
    return [self._get_map(self.item(n).text()) for n in range(0, self.count())]


def _set_QListWidgetAddRemove(self, v):
    """
    Set currently values in QListWidget via re-mapping filter.

    Supply values to be selected as a list.
    """
    block = self.blockSignals(True)
    self.clear()
    self.addItems([unicode(self._set_map(s)) for s in v])
    self.blockSignals(block)
    self.itemAddedOrRemoved.emit()


def _event_QListWidgetAddRemove(self):
    """
    Return current selection changed signal for QListWidget.
    """
    return self.itemAddedOrRemoved


# QColorButton
def _get_QColorButton(self):
    """
    Get current value for QColorButton
    """
    return self.color()


def _set_QColorButton(self, v):
    """
    Set current value for QColorButton
    """
    self.setColor(v)


def _event_QColorButton(self):
    """
    Return value change signal for QColorButton
    """
    return self.colorChanged


# QNoneDoubleSpinBox
def _get_QNoneDoubleSpinBox(self):
    """
    Get current value for QDoubleSpinBox
    """
    return self.value()


def _set_QNoneDoubleSpinBox(self, v):
    """
    Set current value for QDoubleSpinBox
    """
    self.setValue(v)


def _event_QNoneDoubleSpinBox(self):
    """
    Return value change signal for QDoubleSpinBox
    """
    return self.valueChanged


# QCheckTreeWidget
def _get_QCheckTreeWidget(self):
    """
    Get currently checked values in QCheckTreeWidget via re-mapping filter.

    Selected values are returned as a list.
    """
    return [self._get_map(s) for s in self._checked_item_cache]


def _set_QCheckTreeWidget(self, v):
    """
    Set currently checked values in QCheckTreeWidget via re-mapping filter.

    Supply values to be selected as a list.
    """
    if v:
        for s in v:
            f = self.findItems(
                unicode(self._set_map(s)), Qt.MatchExactly | Qt.MatchRecursive
            )
            if f:
                f[0].setCheckState(0, Qt.Checked)


def _event_QCheckTreeWidget(self):
    """
    Return current checked changed signal for QCheckTreeWidget.
    """
    return self.itemCheckedChanged


# QSlider
def _get_QSlider(self):
    """
    Get current value for QSlider
    """
    return self.value()


def _set_QSlider(self, v):
    """
    Set current value for QSlider
    """
    self.setValue(v)


def _event_QSlider(self):
    """
    Return value change signal for QSlider
    """
    return self.valueChanged


# QButtonGroup
def _get_QButtonGroup(self):
    """
    Get a list of (index, checked) tuples for the buttons in the group
    """
    return [(nr, btn.isChecked()) for nr, btn in enumerate(self.buttons())]


def _set_QButtonGroup(self, v):
    """
    Set the states for all buttons in a group from a list of
    (index, checked) tuples
    """
    for idx, state in v:
        self.buttons()[idx].setChecked(state)


def _event_QButtonGroup(self):
    """
    Return button clicked signal for QButtonGroup
    """
    return self.buttonClicked


# QTabWidget
def _get_QTabWidget(self):
    """
    Get the current tabulator index
    """
    return self.currentIndex()


def _set_QTabWidget(self, v):
    """
    Set the current tabulator index
    """
    self.setCurrentIndex(v)


def _event_QTabWidget(self):
    """
    Return currentChanged signal for QTabWidget
    """
    return self.currentChanged


HOOKS = {
    QComboBox: (_get_QComboBox, _set_QComboBox, _event_QComboBox),
    QCheckBox: (_get_QCheckBox, _set_QCheckBox, _event_QCheckBox),
    QAction: (_get_QAction, _set_QAction, _event_QAction),
    QActionGroup: (_get_QActionGroup, _set_QActionGroup, _event_QActionGroup),
    QPushButton: (_get_QPushButton, _set_QPushButton, _event_QPushButton),
    QSpinBox: (_get_QSpinBox, _set_QSpinBox, _event_QSpinBox),
    QDoubleSpinBox: (_get_QDoubleSpinBox, _set_QDoubleSpinBox, _event_QDoubleSpinBox),
    QPlainTextEdit: (_get_QPlainTextEdit, _set_QPlainTextEdit, _event_QPlainTextEdit),
    QLineEdit: (_get_QLineEdit, _set_QLineEdit, _event_QLineEdit),
    QListWidget: (_get_QListWidget, _set_QListWidget, _event_QListWidget),
    QSlider: (_get_QSlider, _set_QSlider, _event_QSlider),
    QButtonGroup: (_get_QButtonGroup, _set_QButtonGroup, _event_QButtonGroup),
    QTabWidget: (_get_QTabWidget, _set_QTabWidget, _event_QTabWidget),
}

default_metadata = {
    "prefer_hidden": False,
    "preferred_handler": None,
    "preferred_map_dict": None,
}


# ConfigManager handles configuration for a given appview
# Supports default values, change signals, export/import from file
# (for workspace saving)
class ConfigManagerBase(QObject):

    # Signals
    # Triggered anytime configuration is changed (refresh)
    updated = Signal(int)

    def __init__(self, defaults=None, *args, **kwargs):
        super(ConfigManagerBase, self).__init__(*args, **kwargs)

        self.mutex = QMutex()
        self.hooks = HOOKS
        self.reset()
        if defaults is None:
            defaults = {}

        # Same mapping as above, used when config not set
        self.defaults = defaults

        self._metadata = {}

    def _get(self, key):
        with QMutexLocker(self.mutex):
            try:
                return self.config[key]
            except:
                return None

    def _get_default(self, key):
        with QMutexLocker(self.mutex):
            try:
                return self.defaults[key]
            except:
                return None

    # Get config
    def get(self, key):
        """
        Get config value for a given key from the config manager.

        Returns the value that matches the supplied key. If the value is
        not set a default value will be returned as set by set_defaults.

        :param key: The configuration key to return a config value for
        :type key: str
        :rtype: Any supported (str, int, bool, list-of-supported-types)
        """
        v = self._get(key)
        if v is not None:
            return v
        return self._get_default(key)

    def get_metadata(self, key):
        with QMutexLocker(self.mutex):
            try:
                return self._metadata[key]
            except KeyError:
                return default_metadata

    def metadata_as_dict(self):
        """
        Get metadata for all items. (If none is set get_metadata returns default)

        TODO: Should both this and as_dict iterate on defaults.keys() | config.keys()?
        """
        metadata_dict = {}
        for k, v in self.defaults.items():
            metadata_dict[k] = self.get_metadata(k)

        return metadata_dict

    def set(self, key, value, trigger_handler=True, trigger_update=True):
        """
        Set config value for a given key in the config manager.

        Set key to value. The optional trigger_update determines whether
        event hooks will fire for this key (and so re-calculation). It is
        useful to suppress these when updating multiple values for example.

        :param key: The configuration key to set
        :type key: str
        :param value: The value to set the configuration key to
        :type value: Any supported
                     (str, int, bool, list-of-supported-types)
        :rtype: bool (success)
        """
        old = self._get(key)
        if old is not None and old == value:
            return False  # Not updating

        # Set value
        self._set(key, value)

        if trigger_handler and key in self.handlers:
            # Trigger handler to update the view
            getter = self.handlers[key].getter
            setter = self.handlers[key].setter

            if setter and getter() != self._get(key):
                setter(self._get(key))

        # Trigger update notification
        if trigger_update:
            self.updated.emit(
                self.eventhooks[key] if key in self.eventhooks else RECALCULATE_ALL
            )

        return True

    # Defaults are used in absence of a set value (use for base settings)
    def set_default(self, key, value, eventhook=RECALCULATE_ALL):
        """
        Set the default value for a given key.

        This will be returned if the value is
        not set in the current config. It is important to include defaults for
        all possible config values for backward compatibility with earlier
        versions of a plugin.

        :param key: The configuration key to set
        :type key: str
        :param value: The value to set the configuration key to
        :type value: Any supported (str, int, bool, list-of-supported-types)
        :param eventhook: Attach either a full recalculation trigger
                          (default), or a view-only recalculation trigger
                          to these values.
        :type eventhook: int RECALCULATE_ALL, RECALCULATE_VIEWS

        """

        self.defaults[key] = value
        self.eventhooks[key] = eventhook
        self.updated.emit(eventhook)

    def set_metadata(self, key, update_dict):
        """
        Set the metadata for a single key.

        TODO: Not sure if this needs to trigger handler etc?

        :param key: Config item key
        :param update_dict: Dict of changes to the default metadata
        """
        with QMutexLocker(self.mutex):
            m = default_metadata.copy()
            for k, v in update_dict.items():
                if k in default_metadata:
                    m[k] = v
            self._metadata[key] = m

    def set_defaults(self, keyvalues, eventhook=RECALCULATE_ALL):
        """
        Set the default value for a set of keys.

        These will be returned if the value is
        not set in the current config. It is important to include defaults for
        all possible config values for backward compatibility with earlier
        versions of a plugin.

        :param keyvalues: A dictionary of keys and values to set as defaults
        :type key: dict
        :param eventhook: Attach either a full recalculation trigger (default),
                          or a view-only recalculation trigger to these values.
        :type eventhook: int RECALCULATE_ALL, RECALCULATE_VIEWS

        """
        for key, value in keyvalues.items():
            self.defaults[key] = value
            self.eventhooks[key] = eventhook

        # Updating the defaults may update the config (if anything
        # without a config value is set by it; should check)
        self.updated.emit(eventhook)

    # Completely replace current config (wipe all other settings)

    def replace(self, keyvalues):
        """
        Completely reset the config with a set of key values.

        Note that this does not wipe handlers or triggers (see reset), it
        simply replaces the values in the config entirely. It is the
        equivalent of unsetting all keys, followed by a set_many.
        Anything not in the supplied keyvalues will revert to default.

        :param keyvalues: A dictionary of keys and values to set as defaults
        :type keyvalues: dict
        :param trigger_update: Flag whether to trigger a config update
                               (+recalculation) after all values are set.

        """
        self.config = {}
        self.set_many(keyvalues)

    def set_many(self, keyvalues, trigger_update=True):
        """
        Set the value of multiple config settings simultaneously.

        This postpones the triggering of the update signal until all values
        are set to prevent excess signals. The trigger_update option can be
        set to False to prevent any update at all.

        :param keyvalues: A dictionary of keys and values to set.
        :type key: dict
        :param trigger_update: Flag whether to trigger a config update
                               (+recalculation) after all values are set.
        :type trigger_update: bool
        """
        has_updated = False
        for k, v in keyvalues.items():
            u = self.set(k, v, trigger_update=False)
            has_updated = has_updated or u

        if has_updated and trigger_update:
            self.updated.emit(RECALCULATE_ALL)

        return has_updated

    def set_many_metadata(self, metadata):
        """
        Set the config manager's metadata attribute. This should be a dict with keys matching each config item for which
        there is metadata. The metadata can include preferred handler, preferred mapper etc.

        :param metadata:
        """
        for key, value in metadata.items():
            self.set_metadata(key, value)

    # HANDLERS

    # Handlers are UI elements (combo, select, checkboxes) that automatically
    # update and updated from the config manager. Allows instantaneous
    # updating on config changes and ensuring that elements remain in sync

    def add_handler(
        self, key, handler=None, mapper=(lambda x: x, lambda x: x), default=None
    ):
        """
        Add a handler (UI element) for a given config key.

        The supplied handler should be a QWidget or QAction through which
        the user can change the config setting. An automatic getter, setter
        and change-event handler is attached which will keep the widget
        and config in sync. The attached handler will default to the correct
        value from the current config.

        An optional mapper may also be provider to handler translation from
        the values shown in the UI and those saved/loaded from the config.

        """
        if key in self.handlers:
            # Already there; so skip must remove first to replace
            return

        # If no handler is supplied, we try to create one either by using either the preferred_handler item in metadata
        # or using a default one based on the type
        if handler is None:
            if self.get_metadata(key)["preferred_handler"] is not None:
                # If there is a preferred handler in the metadata, create one of those. If there is a preferred mapper
                # use that
                handler = self.get_metadata(key)["preferred_handler"]()
                if self.get_metadata(key)["preferred_map_dict"] is not None:
                    mapper = self.get_metadata(key)["preferred_map_dict"]
                    # If we've just created the handler, we need to add the map dict items
                    handler.addItems(
                        self.get_metadata(key)["preferred_map_dict"].keys()
                    )
            # There is not preferred handler, get a default one
            else:
                handler = self.get_default_handler(type(self.get(key)))
            if handler is None:
                return

        # Add map handler for converting displayed values to
        # internal config data
        if isinstance(mapper, (dict, OrderedDict)):
            # By default allow dict types to be used
            mapper = build_dict_mapper(mapper)

        elif isinstance(mapper, list) and isinstance(mapper[0], tuple):
            mapper = build_tuple_mapper(mapper)

        handler._get_map, handler._set_map = mapper

        self.handlers[key] = handler

        # Look for class in hooks and add getter, setter, updater
        cls = self._get_hook(handler)
        hookg, hooks, hooku = self.hooks[cls]

        handler.getter = types_MethodType(hookg, handler)
        handler.setter = types_MethodType(hooks, handler)
        handler.updater = types_MethodType(hooku, handler)

        logging.debug("Add handler %s for %s" % (type(handler).__name__, key))
        handler_callback = lambda x=None: self.set(
            key, handler.getter(), trigger_handler=False
        )
        handler.updater().connect(handler_callback)

        # Store this so we can issue a specific remove on deletes
        self.handler_callbacks[key] = handler_callback

        # If the key is not in defaults, set the default to match the handler
        if key not in self.defaults:
            if default is None:
                self.set_default(key, handler.getter())
            else:
                self.set_default(key, default)

        # Keep handler and data consistent
        if self._get(key) is not None:
            handler.setter(self._get(key))

        # If the key is in defaults; set the handler to the default state
        # (but don't add to config)
        elif key in self.defaults:
            handler.setter(self.defaults[key])

    def _get_hook(self, handler):
        fst = lambda x: next(x, None)

        cls = fst(x for x in self.hooks.keys() if x == type(handler))
        if cls is None:
            cls = fst(x for x in self.hooks.keys() if isinstance(handler, x))

        if cls is None:
            raise TypeError(
                "No handler-functions available for this widget "
                "type (%s)" % type(handler).__name__
            )
        return cls

    def add_handlers(self, keyhandlers):
        for key, handler in list(keyhandlers.items()):
            self.add_handler(key, handler)

    def remove_handler(self, key):
        if key in self.handlers:
            handler = self.handlers[key]
            handler.updater().disconnect(self.handler_callbacks[key])
            del self.handlers[key]

    def add_hooks(self, key, hooks):
        self.hooks[key] = hooks

    def getXMLConfig(self, root):
        config = et.SubElement(root, "Config")
        for ck, cv in list(self.config.items()):
            co = et.SubElement(config, "ConfigSetting")
            co.set("id", ck)
            t = type(cv).__name__
            co.set("type", type(cv).__name__)
            co = CONVERT_TYPE_TO_XML[t](co, cv)

        return root

    def setXMLConfig(self, root):

        config = {}
        for xconfig in root.findall("Config/ConfigSetting"):
            # id="experiment_control" type="unicode" value="monocyte
            # at intermediate differentiation stage (GDS2430_2)"/>
            if xconfig.get("type") in CONVERT_TYPE_FROM_XML:
                v = CONVERT_TYPE_FROM_XML[xconfig.get("type")](xconfig)
            config[xconfig.get("id")] = v

        self.set_many(config, trigger_update=False)

    def as_dict(self):
        """
        Return the combination of defaults and config as a flat dict
        (so it can be pickled)
        """
        result_dict = {}
        for k, v in self.defaults.items():
            result_dict[k] = self.get(k)

        return result_dict

    def all_as_dict(self):
        """
        Return a dict containing defaults, config, and metadata

        :return:
        """

        return {
            "defaults": self.defaults.copy(),
            "config": self.as_dict(),
            "metadata": self.metadata_as_dict(),
        }

    def get_visible_keys(self):
        return [k for k in self.defaults if not self.get_metadata(k)["prefer_hidden"]]

    @staticmethod
    def get_default_handler(in_type):
        """
        Get a default handler widget based on the input type

        :param in_type:
        :return:
        """
        if in_type == str:
            return QLineEdit()
        elif in_type == float:
            return QDoubleSpinBox()
        elif in_type == bool:
            return QCheckBox()
        elif in_type == int:
            return QSpinBox()
        else:
            return None


class ConfigManager(ConfigManagerBase):

    default_path = "config.json"

    def __init__(self, *args, filename=None, **kwargs):
        super().__init__(*args, **kwargs)

        if filename is not None:
            self.path = filename
            self.load()
        else:
            self.path = self.default_path

    def reset(self):
        """
        Reset the config manager to it's initialised state.

        This clears all values, unsets all defaults and removes all
        handlers, maps, and hooks.
        """
        self.config = {}
        self.handlers = {}
        self.handler_callbacks = {}
        self.defaults = {}
        self.maps = {}
        self.eventhooks = {}

    def _get(self, key):
        with QMutexLocker(self.mutex):
            try:
                return self.config[key]
            except:
                return None

    def _set(self, key, value):
        with QMutexLocker(self.mutex):
            self.config[key] = value

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.set_many(json.load(f))

    def save(self):
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(self.as_dict(), f, indent=4)


class QSettingsManager(ConfigManagerBase):
    def __init__(self, *args, warn_no_app_name=True, **kwargs):
        self.warn_no_app_name = warn_no_app_name
        super().__init__(*args, **kwargs)

    def reset(self):
        """
        Reset the config manager to it's initialised state.

        This initialises QSettings, unsets all defaults and removes all
        handlers, maps, and hooks.
        """
        if self.warn_no_app_name:
            app = QApplication.instance()
            if app.applicationName() == "" or app.organizationName() == "":
                warnings.warn(
                    "QApplication.applicationName and QApplication.orginizationName "
                    "must be set for QSettings to persist."
                )

        self.settings = QSettings()
        self.handlers = {}
        self.handler_callbacks = {}
        self.defaults = {}
        self.maps = {}
        self.eventhooks = {}

    def _get(self, key):
        with QMutexLocker(self.mutex):

            v = self.settings.value(key, None)
            if v is not None:
                if type(v) == QVariant and v.type() == QVariant.Invalid:
                    # Invalid check for Qt4
                    return None

                # Map type to that in defaults: required in case QVariant is a
                # string representation of the actual value
                # (e.g. on Windows Reg)
                vt = type(v)
                if key in self.defaults:
                    dt = type(self.defaults[key])
                    if vt == QVariant:
                        # The target type is a QVariant so munge it
                        # If QVariant (Qt4):
                        type_munge = {
                            int: v.toInt,
                            float: v.toFloat,
                            str: v.toString,
                            unicode: v.toString,
                            bool: v.toBool,
                            list: v.toStringList,
                        }
                        v = type_munge[dt]()
                    elif vt != dt and vt == basestring:
                        # Value is stored as unicode so munge it
                        type_munge = {
                            int: lambda x: int(x),
                            float: lambda x: float(x),
                            str: lambda x: str(x),
                            bool: lambda x: x.lower() == u"true",
                            # other types?
                        }
                        v = type_munge[dt](v)

                    v = dt(v)

                return v

            else:
                return None

    def _set(self, key, value):
        with QMutexLocker(self.mutex):
            self.settings.setValue(key, value)


class ConfigDialog(QtWidgets.QDialog):
    """
    A Dialog class inheriting from QtWidgets.QDialog. This class creates layout from the input config using
    build_config_layout, as well as QDialogButtonBox with Ok and Cancel buttons.
    """

    def __init__(self, config, *args, cols=None, **kwargs):
        if "PyQt5" in sys.modules:
            f = kwargs.pop("f", None)
            if f is not None:
                kwargs["flags"] = f
        elif "PySide2" in sys.modules:
            flags = kwargs.pop("flags", None)
            if flags is not None:
                kwargs["f"] = flags

        super().__init__(*args, **kwargs)
        config_dict = config.all_as_dict()
        self.config = ConfigManager(config_dict["defaults"])
        self.config.set_many(config_dict["config"])
        self.config.set_many_metadata(config_dict["metadata"])

        # Build layout from settings
        config_layout_kwargs = {} if cols is None else {"cols": cols}
        config_layout = build_config_layout(self.config, **config_layout_kwargs)

        # Create a button box for the dialog
        if "PyQt6" in sys.modules:
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Reset
                | QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
        else:
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Reset
                | QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel
            )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # QDialogButtonBox places Reset after Ok and Cancel
        button_box.buttons()[2].setText("Reset to Defaults")
        button_box.buttons()[2].clicked.connect(self.show_confirm_reset_dialog)

        # Place everything in a layout in the dialog
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(config_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def show_confirm_reset_dialog(self):
        message_box = QtWidgets.QMessageBox(
            self, text="Are you sure you want to reset to defaults?"
        )
        message_box.setWindowTitle("Warning")
        if "PyQt6" in sys.modules:
            message_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
        else:
            message_box.setStandardButtons(
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            )
        message_box.buttonClicked.connect(
            lambda button: (
                self.config.set_many(self.config.defaults)
                if "OK" in button.text()
                else None
            )
        )
        message_box.exec()


def build_config_layout(config, cols=2):
    """
    Generate a layout based on the input ConfigManager. The layout consists of a user specified number of columns of
    QFormLayout. In each row of the QFormLayout, the label is the config dict key, and the field is the config handler
    for that key.

    :param config: ConfigManager
    :param cols: Number of columns to use
    :return: QHBoxLayout
    """
    h_layout = QtWidgets.QHBoxLayout()
    forms = [QtWidgets.QFormLayout() for _ in range(cols)]
    for form in forms:
        h_layout.addLayout(form)

    num_items = len(config.get_visible_keys())
    for i, key in enumerate(config.get_visible_keys()):
        # Find which column to put the setting in. Columns are filled equally, with remainder to the left. Each column
        # is filled before proceeding to the next.
        f_index = 0
        for j in range(cols):
            if (i + 1) <= ceil((j + 1) * num_items / cols):
                f_index = j
                break

        # Get the handler widget for the key
        if key in config.handlers:
            # If we've already defined a handler, use that
            input_widget = config.handlers[key]
        else:
            # Otherwise, try to add a handler. If it fails, skip this row
            config.add_handler(key)
            if key not in config.handlers:
                continue
            else:
                input_widget = config.handlers[key]

        label = QtWidgets.QLabel(key)
        forms[f_index].addRow(label, input_widget)

    return h_layout
