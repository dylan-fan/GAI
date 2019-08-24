# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-selection-meta.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='feature-selection-meta.proto',
  package='com.webank.ai.fate.common.mlmodel.buffer',
  syntax='proto3',
  serialized_pb=_b('\n\x1c\x66\x65\x61ture-selection-meta.proto\x12(com.webank.ai.fate.common.mlmodel.buffer\"\x87\x04\n\x14\x46\x65\x61tureSelectionMeta\x12\x16\n\x0e\x66ilter_methods\x18\x01 \x03(\t\x12\x12\n\nlocal_only\x18\x02 \x01(\x08\x12\x0c\n\x04\x63ols\x18\x03 \x03(\t\x12N\n\x0bunique_meta\x18\x04 \x01(\x0b\x32\x39.com.webank.ai.fate.common.mlmodel.buffer.UniqueValueMeta\x12U\n\riv_value_meta\x18\x05 \x01(\x0b\x32>.com.webank.ai.fate.common.mlmodel.buffer.IVValueSelectionMeta\x12_\n\x12iv_percentile_meta\x18\x06 \x01(\x0b\x32\x43.com.webank.ai.fate.common.mlmodel.buffer.IVPercentileSelectionMeta\x12S\n\x08\x63oe_meta\x18\x07 \x01(\x0b\x32\x41.com.webank.ai.fate.common.mlmodel.buffer.CoeffOfVarSelectionMeta\x12X\n\x0coutlier_meta\x18\x08 \x01(\x0b\x32\x42.com.webank.ai.fate.common.mlmodel.buffer.OutlierColsSelectionMeta\"\x1e\n\x0fUniqueValueMeta\x12\x0b\n\x03\x65ps\x18\x01 \x01(\x01\"/\n\x14IVValueSelectionMeta\x12\x17\n\x0fvalue_threshold\x18\x01 \x01(\x01\"9\n\x19IVPercentileSelectionMeta\x12\x1c\n\x14percentile_threshold\x18\x01 \x01(\x01\"2\n\x17\x43oeffOfVarSelectionMeta\x12\x17\n\x0fvalue_threshold\x18\x01 \x01(\x01\"G\n\x18OutlierColsSelectionMeta\x12\x12\n\npercentile\x18\x01 \x01(\x01\x12\x17\n\x0fupper_threshold\x18\x02 \x01(\x01\x42\x1b\x42\x19\x46\x65\x61tureSelectionMetaProtob\x06proto3')
)




_FEATURESELECTIONMETA = _descriptor.Descriptor(
  name='FeatureSelectionMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filter_methods', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.filter_methods', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='local_only', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.local_only', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cols', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.cols', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='unique_meta', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.unique_meta', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='iv_value_meta', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.iv_value_meta', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='iv_percentile_meta', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.iv_percentile_meta', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='coe_meta', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.coe_meta', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='outlier_meta', full_name='com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta.outlier_meta', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=75,
  serialized_end=594,
)


_UNIQUEVALUEMETA = _descriptor.Descriptor(
  name='UniqueValueMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.UniqueValueMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='eps', full_name='com.webank.ai.fate.common.mlmodel.buffer.UniqueValueMeta.eps', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=596,
  serialized_end=626,
)


_IVVALUESELECTIONMETA = _descriptor.Descriptor(
  name='IVValueSelectionMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.IVValueSelectionMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value_threshold', full_name='com.webank.ai.fate.common.mlmodel.buffer.IVValueSelectionMeta.value_threshold', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=628,
  serialized_end=675,
)


_IVPERCENTILESELECTIONMETA = _descriptor.Descriptor(
  name='IVPercentileSelectionMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.IVPercentileSelectionMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='percentile_threshold', full_name='com.webank.ai.fate.common.mlmodel.buffer.IVPercentileSelectionMeta.percentile_threshold', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=677,
  serialized_end=734,
)


_COEFFOFVARSELECTIONMETA = _descriptor.Descriptor(
  name='CoeffOfVarSelectionMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.CoeffOfVarSelectionMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value_threshold', full_name='com.webank.ai.fate.common.mlmodel.buffer.CoeffOfVarSelectionMeta.value_threshold', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=736,
  serialized_end=786,
)


_OUTLIERCOLSSELECTIONMETA = _descriptor.Descriptor(
  name='OutlierColsSelectionMeta',
  full_name='com.webank.ai.fate.common.mlmodel.buffer.OutlierColsSelectionMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='percentile', full_name='com.webank.ai.fate.common.mlmodel.buffer.OutlierColsSelectionMeta.percentile', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='upper_threshold', full_name='com.webank.ai.fate.common.mlmodel.buffer.OutlierColsSelectionMeta.upper_threshold', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=788,
  serialized_end=859,
)

_FEATURESELECTIONMETA.fields_by_name['unique_meta'].message_type = _UNIQUEVALUEMETA
_FEATURESELECTIONMETA.fields_by_name['iv_value_meta'].message_type = _IVVALUESELECTIONMETA
_FEATURESELECTIONMETA.fields_by_name['iv_percentile_meta'].message_type = _IVPERCENTILESELECTIONMETA
_FEATURESELECTIONMETA.fields_by_name['coe_meta'].message_type = _COEFFOFVARSELECTIONMETA
_FEATURESELECTIONMETA.fields_by_name['outlier_meta'].message_type = _OUTLIERCOLSSELECTIONMETA
DESCRIPTOR.message_types_by_name['FeatureSelectionMeta'] = _FEATURESELECTIONMETA
DESCRIPTOR.message_types_by_name['UniqueValueMeta'] = _UNIQUEVALUEMETA
DESCRIPTOR.message_types_by_name['IVValueSelectionMeta'] = _IVVALUESELECTIONMETA
DESCRIPTOR.message_types_by_name['IVPercentileSelectionMeta'] = _IVPERCENTILESELECTIONMETA
DESCRIPTOR.message_types_by_name['CoeffOfVarSelectionMeta'] = _COEFFOFVARSELECTIONMETA
DESCRIPTOR.message_types_by_name['OutlierColsSelectionMeta'] = _OUTLIERCOLSSELECTIONMETA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FeatureSelectionMeta = _reflection.GeneratedProtocolMessageType('FeatureSelectionMeta', (_message.Message,), dict(
  DESCRIPTOR = _FEATURESELECTIONMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.FeatureSelectionMeta)
  ))
_sym_db.RegisterMessage(FeatureSelectionMeta)

UniqueValueMeta = _reflection.GeneratedProtocolMessageType('UniqueValueMeta', (_message.Message,), dict(
  DESCRIPTOR = _UNIQUEVALUEMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.UniqueValueMeta)
  ))
_sym_db.RegisterMessage(UniqueValueMeta)

IVValueSelectionMeta = _reflection.GeneratedProtocolMessageType('IVValueSelectionMeta', (_message.Message,), dict(
  DESCRIPTOR = _IVVALUESELECTIONMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.IVValueSelectionMeta)
  ))
_sym_db.RegisterMessage(IVValueSelectionMeta)

IVPercentileSelectionMeta = _reflection.GeneratedProtocolMessageType('IVPercentileSelectionMeta', (_message.Message,), dict(
  DESCRIPTOR = _IVPERCENTILESELECTIONMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.IVPercentileSelectionMeta)
  ))
_sym_db.RegisterMessage(IVPercentileSelectionMeta)

CoeffOfVarSelectionMeta = _reflection.GeneratedProtocolMessageType('CoeffOfVarSelectionMeta', (_message.Message,), dict(
  DESCRIPTOR = _COEFFOFVARSELECTIONMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.CoeffOfVarSelectionMeta)
  ))
_sym_db.RegisterMessage(CoeffOfVarSelectionMeta)

OutlierColsSelectionMeta = _reflection.GeneratedProtocolMessageType('OutlierColsSelectionMeta', (_message.Message,), dict(
  DESCRIPTOR = _OUTLIERCOLSSELECTIONMETA,
  __module__ = 'feature_selection_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.common.mlmodel.buffer.OutlierColsSelectionMeta)
  ))
_sym_db.RegisterMessage(OutlierColsSelectionMeta)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('B\031FeatureSelectionMetaProto'))
# @@protoc_insertion_point(module_scope)