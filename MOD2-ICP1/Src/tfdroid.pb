node {

  name: "I"

  op: "Placeholder"

  attr {

    key: "dtype"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "shape"

    value {

      shape {

        dim {

          size: -1

        }

        dim {

          size: 3

        }

      }

    }

  }

}

node {

  name: "zeros"

  op: "Const"

  attr {

    key: "dtype"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_FLOAT

        tensor_shape {

          dim {

            size: 3

          }

          dim {

            size: 2

          }

        }

        float_val: 0.0

      }

    }

  }

}

node {

  name: "W"

  op: "VariableV2"

  attr {

    key: "container"

    value {

      s: ""

    }

  }

  attr {

    key: "dtype"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "shape"

    value {

      shape {

        dim {

          size: 3

        }

        dim {

          size: 2

        }

      }

    }

  }

  attr {

    key: "shared_name"

    value {

      s: ""

    }

  }

}

node {

  name: "W/Assign"

  op: "Assign"

  input: "W"

  input: "zeros"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@W"

      }

    }

  }

  attr {

    key: "use_locking"

    value {

      b: true

    }

  }

  attr {

    key: "validate_shape"

    value {

      b: true

    }

  }

}

node {

  name: "W/read"

  op: "Identity"

  input: "W"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@W"

      }

    }

  }

}

node {

  name: "zeros_1"

  op: "Const"

  attr {

    key: "dtype"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_FLOAT

        tensor_shape {

          dim {

            size: 2

          }

        }

        float_val: 0.0

      }

    }

  }

}

node {

  name: "b"

  op: "VariableV2"

  attr {

    key: "container"

    value {

      s: ""

    }

  }

  attr {

    key: "dtype"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "shape"

    value {

      shape {

        dim {

          size: 2

        }

      }

    }

  }

  attr {

    key: "shared_name"

    value {

      s: ""

    }

  }

}

node {

  name: "b/Assign"

  op: "Assign"

  input: "b"

  input: "zeros_1"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@b"

      }

    }

  }

  attr {

    key: "use_locking"

    value {

      b: true

    }

  }

  attr {

    key: "validate_shape"

    value {

      b: true

    }

  }

}

node {

  name: "b/read"

  op: "Identity"

  input: "b"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@b"

      }

    }

  }

}

node {

  name: "MatMul"

  op: "MatMul"

  input: "I"

  input: "W/read"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "transpose_a"

    value {

      b: false

    }

  }

  attr {

    key: "transpose_b"

    value {

      b: false

    }

  }

}

node {

  name: "add"

  op: "AddV2"

  input: "MatMul"

  input: "b/read"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

}

node {

  name: "O"

  op: "Relu"

  input: "add"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

}

node {

  name: "save/filename/input"

  op: "Const"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_STRING

        tensor_shape {

        }

        string_val: "model"

      }

    }

  }

}

node {

  name: "save/filename"

  op: "PlaceholderWithDefault"

  input: "save/filename/input"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "shape"

    value {

      shape {

      }

    }

  }

}

node {

  name: "save/Const"

  op: "PlaceholderWithDefault"

  input: "save/filename"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "shape"

    value {

      shape {

      }

    }

  }

}

node {

  name: "save/SaveV2/tensor_names"

  op: "Const"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_STRING

        tensor_shape {

          dim {

            size: 2

          }

        }

        string_val: "W"

        string_val: "b"

      }

    }

  }

}

node {

  name: "save/SaveV2/shape_and_slices"

  op: "Const"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_STRING

        tensor_shape {

          dim {

            size: 2

          }

        }

        string_val: ""

        string_val: ""

      }

    }

  }

}

node {

  name: "save/SaveV2"

  op: "SaveV2"

  input: "save/Const"

  input: "save/SaveV2/tensor_names"

  input: "save/SaveV2/shape_and_slices"

  input: "W"

  input: "b"

  attr {

    key: "dtypes"

    value {

      list {

        type: DT_FLOAT

        type: DT_FLOAT

      }

    }

  }

}

node {

  name: "save/control_dependency"

  op: "Identity"

  input: "save/Const"

  input: "^save/SaveV2"

  attr {

    key: "T"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@save/Const"

      }

    }

  }

}

node {

  name: "save/RestoreV2/tensor_names"

  op: "Const"

  device: "/device:CPU:0"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_STRING

        tensor_shape {

          dim {

            size: 2

          }

        }

        string_val: "W"

        string_val: "b"

      }

    }

  }

}

node {

  name: "save/RestoreV2/shape_and_slices"

  op: "Const"

  device: "/device:CPU:0"

  attr {

    key: "dtype"

    value {

      type: DT_STRING

    }

  }

  attr {

    key: "value"

    value {

      tensor {

        dtype: DT_STRING

        tensor_shape {

          dim {

            size: 2

          }

        }

        string_val: ""

        string_val: ""

      }

    }

  }

}

node {

  name: "save/RestoreV2"

  op: "RestoreV2"

  input: "save/Const"

  input: "save/RestoreV2/tensor_names"

  input: "save/RestoreV2/shape_and_slices"

  device: "/device:CPU:0"

  attr {

    key: "dtypes"

    value {

      list {

        type: DT_FLOAT

        type: DT_FLOAT

      }

    }

  }

}

node {

  name: "save/Assign"

  op: "Assign"

  input: "W"

  input: "save/RestoreV2"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@W"

      }

    }

  }

  attr {

    key: "use_locking"

    value {

      b: true

    }

  }

  attr {

    key: "validate_shape"

    value {

      b: true

    }

  }

}

node {

  name: "save/Assign_1"

  op: "Assign"

  input: "b"

  input: "save/RestoreV2:1"

  attr {

    key: "T"

    value {

      type: DT_FLOAT

    }

  }

  attr {

    key: "_class"

    value {

      list {

        s: "loc:@b"

      }

    }

  }

  attr {

    key: "use_locking"

    value {

      b: true

    }

  }

  attr {

    key: "validate_shape"

    value {

      b: true

    }

  }

}

node {

  name: "save/restore_all"

  op: "NoOp"

  input: "^save/Assign"

  input: "^save/Assign_1"

}

node {

  name: "init"

  op: "NoOp"

  input: "^W/Assign"

  input: "^b/Assign"

}

versions {

  producer: 119

}